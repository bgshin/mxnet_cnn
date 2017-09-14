import numpy as np
import sys
import mmap
import ctypes
import posix_ipc
from _multiprocessing import address_of_buffer
from string import ascii_letters, digits


valid_chars = frozenset("-_. %s%s" % (ascii_letters, digits))

class ShmemBufferWrapper(object):
    def __init__(self, tag, size, create=True):
        self._mem = None
        self._map = None
        self._owner = create
        self.size = size

        assert 0 <= size < sys.maxint
        flag = (0, posix_ipc.O_CREX)[create]
        if create:
            self._mem = posix_ipc.SharedMemory(tag, flags=flag, size=size)
        else:
            self._mem = posix_ipc.SharedMemory(tag, flags=flag, size=0)
        self._map = mmap.mmap(self._mem.fd, self._mem.size)
        self._mem.close_fd()


    def get_address(self):
        addr, size = address_of_buffer(self._map)
        assert size >= self.size
        return addr


    def __del__(self):
        if self._map is not None:
            self._map.close()
        if self._mem is not None and self._owner:
            self._mem.unlink()

def ShmemRawArray(type_, size_or_initializer, tag, create=True):
    assert frozenset(tag).issubset(valid_chars)
    if tag[0] != "/":
        tag = "/%s" % (tag,)

    if isinstance(size_or_initializer, int):
        type_ = type_ * size_or_initializer
    else:
        type_ = type_ * len(size_or_initializer)

    buffer = ShmemBufferWrapper(tag, ctypes.sizeof(type_), create=create)
    obj = type_.from_address(buffer.get_address())
    obj._buffer = buffer

    if not isinstance(size_or_initializer, int):
        obj.__init__(*size_or_initializer)

    return obj

def np_type_id_to_ctypes(dtype):
    type_id = None
    if hasattr(np, 'float') and dtype == np.float:
        type_id = ctypes.c_int64
        return type_id
    if hasattr(np, 'float16') and dtype == np.float16:
        type_id = ctypes.c_int16
        return type_id
    if hasattr(np, 'float32') and dtype == np.float32:
        type_id = ctypes.c_int32
        return type_id
    if hasattr(np, 'float64') and dtype == np.float64:
        type_id = ctypes.c_int64
        return type_id
    if hasattr(np, 'float128') and dtype == np.float128:
        type_id = (ctypes.c_int64 * 2)
        return type_id
    if hasattr(np, 'int') and dtype == np.int:
        type_id = ctypes.c_int64
        return type_id
    if hasattr(np, 'uint8') and dtype == np.uint8:
        type_id = ctypes.c_byte
        return type_id
    if hasattr(np, 'int8') and dtype == np.int8:
        type_id = ctypes.c_byte
        return type_id
    if hasattr(np, 'uint16') and dtype == np.uint16:
        type_id = ctypes.c_int16
        return type_id
    if hasattr(np, 'int16') and dtype == np.int16:
        type_id = ctypes.c_int16
        return type_id
    if hasattr(np, 'uint32') and dtype == np.uint32:
        type_id = ctypes.c_int32
        return type_id
    if hasattr(np, 'int32') and dtype == np.int32:
        type_id = ctypes.c_int32
        return type_id
    if hasattr(np, 'uint64') and dtype == np.uint64:
        type_id = ctypes.c_int64
        return type_id
    if hasattr(np, 'int64') and dtype == np.int64:
        type_id = ctypes.c_int64
        return type_id
    if hasattr(np, 'complex') and dtype == np.complex:
        type_id = (ctypes.c_int64 * 2)
        return type_id
    if hasattr(np, 'complex64') and dtype == np.complex64:
        type_id = (ctypes.c_int64)
        return type_id
    if hasattr(np, 'intc') and dtype == np.intc:
        type_id = (ctypes.c_int)
        return type_id
    if hasattr(np, 'intp') and dtype == np.intp:
        type_id = (ctypes.c_ssize_t)
        return type_id
    if hasattr(np, 'bool') and dtype == np.bool:
        type_id = (ctypes.c_byte)
        return type_id
    raise Exception('No matching data type!')

class SharedNPArray:
    def __init__(self, shape, dtype, tag=None, create=True):
        type_id = np_type_id_to_ctypes(dtype)
        self.tag = tag
        self.__shared = ShmemRawArray(type_id, np.product(shape), tag, create=create)
        self.__np_array = np.frombuffer(self.__shared, dtype=dtype).reshape(shape)

    def __getattr__(self, name):
        return self.__np_array.__getattribute__(name)

    def copyto(self, nparray):
        np.copyto(self.__np_array, nparray)

