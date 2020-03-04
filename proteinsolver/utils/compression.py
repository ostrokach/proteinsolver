import io

import brotli
import pyarrow as pa


def compress(array: pa.Array) -> bytes:
    rb = pa.RecordBatch.from_arrays([array], ["array"])
    buf = io.BytesIO()
    writer = pa.RecordBatchFileWriter(buf, rb.schema)
    writer.write_batch(rb)
    writer.close()
    buf.seek(0)
    return brotli.compress(buf.read())


def decompress(pybytes: bytes) -> pa.Array:
    buf = io.BytesIO()
    buf.write(brotli.decompress(pybytes))
    buf.seek(0)
    reader = pa.RecordBatchFileReader(buf)
    rb = reader.get_batch(0)
    return rb.column(0)
