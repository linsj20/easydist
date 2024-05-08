# TensorField

TensorField is a module used for unified memory management across processes (libraries). In multi-process scenarios, such as with PyTorch, each process has its own private memory pool. Through the CacheAllocator mechanism, memory is reserved as much as possible without being released, and this reserved memory cannot be used by other processes, resulting in significantly reduced memory usage efficiency. The `tfield-server` is a unified memory management process that helps multiple processes manage memory through a shared memory pool.