use std::fmt;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum DType {
    BF16,
    FP32,
    FP8E4M3,
    FP4E2M1,
    FP8E8M0,
    INT32,
    INT64,
    UINT8,
}

impl DType {
    pub fn element_size(&self) -> usize {
        match self {
            DType::BF16 => 2,
            DType::FP32 => 4,
            DType::FP8E4M3 => 1,
            DType::FP4E2M1 => 1,
            DType::FP8E8M0 => 1,
            DType::INT32 => 4,
            DType::INT64 => 8,
            DType::UINT8 => 1,
        }
    }

    pub fn dlpack_type_code(&self) -> (u8, u8, u16) {
        match self {
            DType::BF16 => (4, 16, 1),
            DType::FP32 => (2, 32, 1),
            DType::FP8E4M3 => (10, 8, 1),
            DType::FP4E2M1 => (17, 4, 2),
            DType::FP8E8M0 => (14, 8, 1),
            DType::INT32 => (0, 32, 1),
            DType::INT64 => (0, 64, 1),
            DType::UINT8 => (1, 8, 1),
        }
    }

    pub fn from_safetensors(name: &str) -> Option<Self> {
        match name {
            "BF16" | "bf16" => Some(DType::BF16),
            "F32" | "fp32" | "float32" => Some(DType::FP32),
            "F8_E4M3" | "fp8e4m3" | "float8_e4m3fn" => Some(DType::FP8E4M3),
            "F8_E8M0" | "fp8e8m0" | "float8_e8m0fnu" => Some(DType::FP8E8M0),
            "I8" | "int8" => Some(DType::UINT8),
            "I32" | "int32" => Some(DType::INT32),
            "I64" | "int64" => Some(DType::INT64),
            "U8" | "uint8" => Some(DType::UINT8),
            _ => None,
        }
    }

    pub fn from_config_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "fp4" => Some(DType::FP4E2M1),
            "fp8" => Some(DType::FP8E4M3),
            "bf16" | "bfloat16" => Some(DType::BF16),
            "fp32" | "float32" => Some(DType::FP32),
            _ => None,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DType::BF16 => write!(f, "bf16"),
            DType::FP32 => write!(f, "fp32"),
            DType::FP8E4M3 => write!(f, "fp8e4m3"),
            DType::FP4E2M1 => write!(f, "fp4e2m1"),
            DType::FP8E8M0 => write!(f, "fp8e8m0"),
            DType::INT32 => write!(f, "i32"),
            DType::INT64 => write!(f, "i64"),
            DType::UINT8 => write!(f, "u8"),
        }
    }
}
