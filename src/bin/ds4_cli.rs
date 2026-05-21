use anyhow::Result;
use ds4rs::{init_tvm_runtime, KernelRegistry, Transformer};
use ds4rs::generate::{GenerateConfig, Generator};
use ds4rs::tokenizer::{ChatMessage, Tokenizer, encode_chat, parse_assistant_response};
use std::io::{self, Write};
use std::sync::Arc;
use std::time::Instant;

const MODEL_DIR: &str = "/models";

fn main() -> Result<()> {
    eprintln!("ds4rs v{}", env!("CARGO_PKG_VERSION"));

    // 初始化 CUDA 设备
    let device = cudarc::driver::CudaContext::new(0)?;

    // 禁用 CUDA 内存池的释放延迟，确保释放的内存立即可用
    // RTX 5060 Ti 16GB 显存紧张，不能让内存池缓存已释放的块
    unsafe {
        let mut pool: cudarc::driver::sys::CUmemoryPool = std::ptr::null_mut();
        let result = cudarc::driver::sys::cuDeviceGetMemPool(
            &mut pool,
            0,
        );
        if result == cudarc::driver::sys::CUresult::CUDA_SUCCESS {
            let mut disable: usize = 0;
            cudarc::driver::sys::cuMemPoolSetAttribute(
                pool,
                cudarc::driver::sys::CUmemPool_attribute::CU_MEMPOOL_ATTR_RELEASE_THRESHOLD,
                &mut disable as *mut usize as *mut std::ffi::c_void,
            );
        }
    }

    eprintln!("[init] CUDA device ready");

    // 初始化 TVM 运行时（加载 TileLang 编译的 .so 内核）
    let runtime = init_tvm_runtime()?;
    eprintln!("[init] TVM runtime loaded: {:?}", runtime.lib_path());

    let kernels = Arc::new(KernelRegistry::new(runtime));

    // 加载 TileLang 预编译内核
    let kernel_dir = "/workspace/tilelang/build";
    if std::path::Path::new(kernel_dir).exists() {
        match kernels.load_dir(kernel_dir) {
            Ok(n) => eprintln!("[init] Loaded {} TileLang kernels", n),
            Err(e) => eprintln!("[init] Warning: kernel load failed: {}", e),
        }
    }

    // 加载模型权重到 GPU
    // max_seqlen 控制 KV cache 预分配大小，影响显存占用
    // RTX 5060 Ti 16GB: 1024 约需 ~12GB，2048 可能 OOM
    let max_seqlen: usize = std::env::var("DS4RS_MAX_SEQLEN")
        .map(|s| s.parse().unwrap_or(1024))
        .unwrap_or(1024);
    eprintln!("[init] Loading model from {} (max_seqlen={}) ...", MODEL_DIR, max_seqlen);
    let model = Transformer::load(MODEL_DIR, device, 1, max_seqlen, kernels)?;
    eprintln!(
        "[init] Model loaded: {} layers, vocab={}, hidden={}",
        model.config.num_hidden_layers,
        model.config.vocab_size,
        model.config.hidden_size
    );

    // 加载分词器
    eprintln!("[init] Loading tokenizer ...");
    let tokenizer = Tokenizer::from_dir(MODEL_DIR)?;
    eprintln!(
        "[init] Tokenizer ready: vocab_size={}, bos={}, eos={}",
        tokenizer.vocab_size(),
        tokenizer.bos_id(),
        tokenizer.eos_id()
    );

    // 配置生成参数
    let gen_config = GenerateConfig {
        max_new_tokens: 64,
        temperature: 0.6,
        top_p: 0.9,
        repetition_penalty: 1.1,
        repetition_window: 64,
    };

    let mut generator = Generator::new(model, tokenizer, gen_config);

    // 多轮对话消息历史
    let mut messages: Vec<ChatMessage> = Vec::new();

    eprintln!("\nDeepSeek V4 Flash — type /exit to quit, /clear to reset context\n");

    loop {
        eprint!(">>> ");
        io::stderr().flush()?;
        let mut input = String::new();
        if io::stdin().read_line(&mut input).is_err() || input.is_empty() {
            break;
        }
        let prompt = input.trim();

        if prompt == "/exit" {
            break;
        }
        if prompt == "/clear" {
            messages.clear();
            generator.reset();
            eprintln!("[context cleared]");
            continue;
        }
        if prompt.is_empty() {
            continue;
        }

        messages.push(ChatMessage::user(prompt));

        // 编码对话历史为模型输入格式
        let chat_text = encode_chat(&messages, false);
        let prompt_tokens = generator.tokenizer().encode(&chat_text, true)?;
        eprintln!("[debug] chat_text: {:?}", chat_text);
        eprintln!("[debug] prompt_tokens: {:?}", prompt_tokens);

        eprintln!("[prefill {} tokens, generating ...]", prompt_tokens.len());

        // 计时：测量生成速度 (tokens/s)
        let gen_start = Instant::now();

        // 流式回调：逐 token 输出并计数
        let callback = |text: &str| {
            print!("{}", text);
            io::stdout().flush().ok();
        };

        match generator.generate(&prompt_tokens, Some(&callback)) {
            Ok(completion_tokens) => {
                let gen_elapsed = gen_start.elapsed();
                // 计算生成 token 数（不含 prompt）
                let token_count = completion_tokens.len() - prompt_tokens.len();
                let tokens_per_sec = token_count as f64 / gen_elapsed.as_secs_f64();

                println!();
                eprintln!(
                    "\n[stats] {} tokens in {:.2}s → {:.1} t/s",
                    token_count,
                    gen_elapsed.as_secs_f64(),
                    tokens_per_sec
                );

                // 解码助手回复并加入对话历史
                let completion_text = if let Some(eos_pos) = completion_tokens.iter().position(|&t| t == generator.tokenizer().eos_id()) {
                    let decoded = generator.tokenizer().decode(&completion_tokens[..eos_pos])?;
                    parse_assistant_response(&decoded)
                } else {
                    generator.tokenizer().decode(&completion_tokens)?
                };

                messages.push(ChatMessage::assistant(&completion_text));
            }
            Err(e) => {
                eprintln!("\n[error] generation failed: {}", e);
            }
        }
    }

    Ok(())
}
