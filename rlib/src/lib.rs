unsafe extern "C" {
    fn asm_add_one(x: i64) -> i64;
}

pub fn add_one(x: i64) -> i64 {
    unsafe { asm_add_one(x) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn it_works() {
        let result = add_one(41);
        assert_eq!(result, 42);
    }
}
