use core::sync::atomic::{AtomicBool, Ordering};

static QEMU_TEST_ENV: AtomicBool = AtomicBool::new(false);

pub fn is_qemu_test_env() -> bool {
    QEMU_TEST_ENV.load(Ordering::Acquire)
}

pub fn set_qemu_test_env() {
    QEMU_TEST_ENV.store(true, Ordering::Release)
}
