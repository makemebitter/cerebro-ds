# Copyright 2020 Yuhao Zhang and Arun Kumar. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from collections import namedtuple
import struct
# import bitstruct as bitstruct
import pandas as pd
from cerebro_gpdb.utils import logs
import socket
import numpy as np
import ctypes
import glob
import sys
IS_PYTHON3 = sys.version_info.major >= 3
HEAP_HASNULL = 0x0001
HEAP_NATTS_MASK = 0x07FF
TYPALIGN_INT = 'i'
VARHDRSZ_EXTERNAL = 4
SIZE_OF_VARATT_EXTERNAL = 16
INDEPENDENT_VAR_OFF = 4
FULL_PD_UPPER = 64
# AD_HOC_BYTES = b'0\x08\x00\x00\x00\x00\x00\x00h'
BLOCK_SIZE = 32768
BLCKSZ = BLOCK_SIZE
PAGE_HEADER_LEN = 24
ITEM_IDENTIFIER_LEN = 4
VARHDRSZ = 4
SIZE_OF_ITEMIDDATA = ITEM_IDENTIFIER_LEN
ITEM_HEADER_LEN = 23
MAXIMUM_ALIGNOF = 8
SIZE_OF_PGLZ_HEADER = 8
CHUNK_ID_AND_CHUNK_SEQ_LEN = 8
TOAST_MAX_CHUNK_SIZE = 8140
SIZE_OF_PAGEHEADERDATA = PAGE_HEADER_LEN + ITEM_IDENTIFIER_LEN
PGLZ_IMPLEMENTATION = 'py'  # C implementation not working as of now
# lib = ctypes.CDLL(os.path.abspath("./pg_lzcompress.so"))
# pglz_decompress_c = wrap_function(lib, 'pglz_decompress', None,
# [ctypes.c_char_p, ctypes.c_char_p])


def wrap_function(lib, funcname, restype, argtypes):
    """Simplify wrapping ctypes functions"""
    func = lib.__getattr__(funcname)
    func.restype = restype
    func.argtypes = argtypes
    return func


def SET_VARSIZE(bytea, header):
    bytea[:4] = header


def att_align_nominal(off):
    cur_offset = off
    LEN = cur_offset
    ALIGNOF_INT = 4
    ALIGNVAL = ALIGNOF_INT
    return (LEN + ALIGNVAL - 1) & ~(ALIGNVAL - 1)


def TYPEALIGN(ALIGNVAL, LEN):
    return (LEN + ALIGNVAL - 1) & ~(ALIGNVAL - 1)


def MAX_ALIGN(LEN):
    return TYPEALIGN(MAXIMUM_ALIGNOF, (LEN))


def VARSIZE_1B_E():
    return SIZE_OF_VARATT_EXTERNAL + VARHDRSZ_EXTERNAL


def VARSIZE_1B(bytea):
    header = get_1b_header(bytea)
    return (header) & 0x7F


# def VARATT_IS_EXTERNAL(byte):
#     return byte == 0x80


def VARDATA(bytea):
    return bytea[4:]


def get_1b_header(bytea):
    if isinstance(bytea, int):
        return bytea
    else:
        return struct.unpack('@B', bytea[0:1])[0]


def VARATT_IS_1B(bytea):
    header = get_1b_header(bytea)
    return (header & 0x80) == 0x80


def VARATT_IS_4B_U(bytea):
    header = get_1b_header(bytea)
    return (header & 0xC0) == 0x00


def VARATT_IS_EXTENDED(bytea):
    return not VARATT_IS_4B_U(bytea)

def VARATT_IS_1B_E(bytea):
    header = get_1b_header(bytea)
    return header == 0x80

def VARATT_IS_4B_C(bytea):
    header = get_1b_header(bytea)
    return (header & 0xC0) == 0x40
def VARATT_IS_COMPRESSED(bytea):
    return VARATT_IS_4B_C(bytea)

def VARATT_IS_EXTERNAL(bytea):
    return VARATT_IS_1B_E(bytea)

def VARSIZE(bytea):
    return socket.ntohl(struct.unpack(str('@I'), bytea[:4])[0]) & 0x3FFFFFFF


def GET_VARSIZE_4B_C(total_length):
    length_tmp = (total_length & 0x3FFFFFFF) | 0x40000000
    length_byte = struct.pack('>I', length_tmp)
    return length_byte


def GET_VARSIZE_4B(total_length):
    length_tmp = (total_length & 0x3FFFFFFF)
    length_byte = struct.pack('>I', length_tmp)
    return length_byte


# def item_identifiers_debug(item_identifiers_bytes):
#     for i in range(0, len(item_identifiers_bytes), ITEM_IDENTIFIER_LEN):
#         packed_identifier = item_identifiers_bytes[i:i + ITEM_IDENTIFIER_LEN]
#         print(packed_identifier)
#         lp_off, lp_flags, lp_len = bitstruct.unpack('u15u2u15<',
#                                                     packed_identifier)
#         print(lp_off, lp_flags, lp_len)


def get_num_chunks(ressize):
    return ((ressize - 1) // TOAST_MAX_CHUNK_SIZE) + 1


def expand_input_dims(input_data):
    input_data = np.array(input_data, dtype=np.float32)
    input_data = np.expand_dims(input_data, axis=0)
    return input_data


def np_array_float32(var, var_shape):
    arr = np.frombuffer(var, dtype=np.float32)
    arr.shape = var_shape
    return arr


def np_array_int16(var, var_shape):
    arr = np.frombuffer(var, dtype=np.int16)
    arr.shape = var_shape
    return arr


def deserialize_bytea(attname, dest_data, var_shape):
    if attname == 'dependent_var':
        np_data = np_array_int16(dest_data, var_shape)
    elif attname == 'independent_var':
        np_data = np_array_float32(dest_data, var_shape)
    return np_data


def GET_RAWSIZE_FROM_COMPRESSED(bytea):
    return struct.unpack('@I', bytea[4:8])[0]

def acc(bytea, i):
    return bytea[i:i+1]

def pglz_decompress_py(source, dest):
    varsize_source = VARSIZE(source)
    rawsize_source = GET_RAWSIZE_FROM_COMPRESSED(source)
    sp = 0 + SIZE_OF_PGLZ_HEADER
    srcend = 0 + varsize_source
    dp = 0
    destend = dp + rawsize_source
    while (sp < srcend and dp < destend):
        ctrl = source[sp]
        ctrl = get_1b_header(ctrl)
        sp += 1
        ctrlc = 0
        while(ctrlc < 8 and sp < srcend):
            if (ctrl & 1):
                length = (get_1b_header(source[sp]) & 0x0f) + 3
                off = ((
                    get_1b_header(source[sp]) & 0xf0) << 4) | get_1b_header(
                    source[sp+1])
                sp += 2
                if (length == 18):
                    length += get_1b_header(source[sp])
                    sp += 1
                if (dp + length > destend):
                    dp += length
                    break
                while (length):
                    length -= 1
                    dest[dp] = dest[dp - off]
                    dp += 1
            else:
                if (dp >= destend):
                    break
                dest[dp] = source[sp]
                dp += 1
                sp += 1

            ctrl >>= 1
            ctrlc += 1
    if (dp != destend or sp != srcend):
        raise Exception("compressed data is corrupt")
    return dest


def pglz_decompress(source, dest, implementation):
    if implementation == 'py':
        return pglz_decompress_py(source, dest)


def pre_alloc_dest(bytea, implementation):
    dest_data_size = struct.unpack('@I', bytea[4:8])[0]
#     dest_size = dest_data_size + VARHDRSZ
#     dest_header = GET_VARSIZE_4B(dest_size)
#     SET_VARSIZE(dest, dest_header)
#     dest_data = VARDATA(dest)
    if implementation == 'py':
        dest_data = bytearray(dest_data_size)
    elif implementation == 'c':
        dest_data = ctypes.create_string_buffer(dest_data_size)

    return dest_data, dest_data_size


def deserialize_page_header(page_bytes):
    deserialized_header = struct.unpack(
        str('@qHHHHHHI'), page_bytes[:PAGE_HEADER_LEN])
    deserialized_header = list(deserialized_header)
    deserialized_header[-2] &= 0x00FF
    header = PageHeader(*deserialized_header)
    item_identifiers_begin = PAGE_HEADER_LEN
    item_identifiers_end = getattr(header, 'pd_lower')
    item_identifiers_len = item_identifiers_end - item_identifiers_begin
    if (item_identifiers_len % ITEM_IDENTIFIER_LEN) != 0:
        raise Exception("""Item identifier length do not match.
                        Must be a multiply of {}.""".format(
            ITEM_IDENTIFIER_LEN))
    item_identifiers_bytes = page_bytes[
        item_identifiers_begin:item_identifiers_end]
    item_num = item_identifiers_len // ITEM_IDENTIFIER_LEN
    return header, item_identifiers_bytes, item_num


def deserialize_item_header(item):
    deserialized_item_header = struct.unpack(
        str('@IIIHHHHHB'), item[:ITEM_HEADER_LEN])
    item_header = HeapTupleHeader(*deserialized_item_header)
    return item_header


def extract_tuple(item, item_header, tuple_data_len):
    t_hoff = getattr(item_header, 't_hoff')
    tuple_data_bytea = item[t_hoff:t_hoff+tuple_data_len]
    return tuple_data_bytea


def deserialize_item(page_bytes, packed_identifier):
    assert len(packed_identifier) == ITEM_IDENTIFIER_LEN
    bitlengthes = [15, 2, 15]
#     LSB first
    names = ['lp_len', 'lp_flags', 'lp_off']
    values = []
    bitstring = '{:032b}'.format(struct.unpack('@I', packed_identifier)[0])
    off = 0
    for bitlength, name in zip(bitlengthes, names):
        bitfield = bitstring[off:off + bitlength]
        number = int(bitfield, base=2)
        values.append(number)
        off += bitlength
    values = values[::-1]
    item_identifier = ItemIdentifier(*values)
    lp_off, lp_flags, lp_len = item_identifier
    item = page_bytes[lp_off:lp_off + lp_len]
    item_header = deserialize_item_header(item)
    t_hoff = getattr(item_header, 't_hoff')
    tupdata = item[t_hoff:]
    return tupdata, item_header, item_identifier


# def desirialize_tupdata(tupdata, toast=False):
#     if not toast:
#         length = VARSIZE_1B_E()
#         c = 0
#         n = 4
#         dist_key = struct.unpack('@I', tupdata[c:n])[0]
#         c = n
#         n += length
#         independent_var = struct.unpack('@BBBBiiII', tupdata[c:n])
#         c = n
#         n += length
#         dependent_var = struct.unpack('@BBBBiiII', tupdata[c:n])
#         buffer_id = struct.unpack('@I', tupdata[-4:])[0]
#         independent_var_tuple = Tuple(*independent_var)
#         dependent_var_tuple = Tuple(*dependent_var)
#         return dist_key, independent_var_tuple, dependent_var_tuple, buffer_id
#     else:
#         chunk_id, chunk_seq = struct.unpack(str('@II'), tupdata[:8])
#         chunk_data = tupdata[8:]
#         return chunk_id, chunk_seq, chunk_data
def desirialize_tupdata(tupdata, toast=False):
    if not toast:
        buffer_id = struct.unpack('@I', tupdata[-4:])[0]
        length = VARSIZE_1B_E()
        c = 0
        n = 4
        dist_key = struct.unpack('@I', tupdata[c:n])[0]
        c = n
        n += length
        independent_var = struct.unpack('@BBBBiiII', tupdata[c:n])
        c = n
        n += length
        dep_header = tupdata[c:n]
        buffer_id = struct.unpack('@I', tupdata[-4:])[0]
        dependent_var = struct.unpack('@BBBBiiII', tupdata[c:n])
        if VARATT_IS_EXTENDED(dep_header) and not VARATT_IS_EXTERNAL(dep_header) and VARATT_IS_COMPRESSED(dep_header):
            bytea = tupdata[c:]
            dependent_var = [x for x in dependent_var] + [False, bytea]
        else:
            dependent_var = [x for x in dependent_var] + [True, None]
        independent_var = [x for x in independent_var] + [True, None]
        independent_var_tuple = Tuple(*independent_var)
        dependent_var_tuple = Tuple(*dependent_var)
        return dist_key, independent_var_tuple, dependent_var_tuple, buffer_id
    else:
        chunk_id, chunk_seq = struct.unpack(str('@II'), tupdata[:8])
        chunk_data = tupdata[8:]
        return chunk_id, chunk_seq, chunk_data


def read_page_at(f, pos):
    f.seek(pos)
    page_bytes = f.read(BLOCK_SIZE)
    return page_bytes


def generator_page(filename, toast=False, debug=False):
    seg_files = glob.glob(filename + '.*')
    seg_files.sort()
    seg_files.sort(key=len)
    seg_files = [filename] + seg_files
    for filename in seg_files:
        with open(filename, 'rb') as f:
            curr_pos = 0
            while True:

                page_bytes = read_page_at(f, curr_pos)
                if len(page_bytes) != BLOCK_SIZE:
                    if len(page_bytes) != 0:
                        raise Exception("FATAL")
                    else:
                        break
                if debug:
                    print("PAGE: {}".format(curr_pos))
                curr_pos = f.tell()
                page_header, item_identifiers_bytes, item_num = \
                    deserialize_page_header(page_bytes)
                if toast:
                    pd_upper = getattr(page_header, 'pd_upper')
                    pd_special = getattr(page_header, 'pd_special')
                    assert pd_special == BLOCK_SIZE, \
                        "THERE SHALL NOT BE INDICES"
                    lp_flags = 1
                    lp_off = pd_upper

                    for idx in range(item_num):
                        lp_off = MAX_ALIGN(lp_off)
                        item_header = page_bytes[lp_off:lp_off +
                                                 ITEM_HEADER_LEN]
                        item_header = deserialize_item_header(item_header)
                        t_hoff = getattr(item_header, 't_hoff')
                        tupdata_off = lp_off + t_hoff
                        chunkdata_off = tupdata_off + \
                            CHUNK_ID_AND_CHUNK_SEQ_LEN
                        chunkdata_header = page_bytes[
                            chunkdata_off:chunkdata_off + VARHDRSZ]
                        chunksize = VARSIZE(chunkdata_header)
                        lp_len = t_hoff + CHUNK_ID_AND_CHUNK_SEQ_LEN + \
                            chunksize
                        item_identifier = ItemIdentifier(
                            *(lp_off, lp_flags, lp_len))
                        item = page_bytes[lp_off:lp_off + lp_len]

                        tupdata = item[t_hoff:]
                        chunk_id, chunk_seq, chunk_data = desirialize_tupdata(
                            tupdata, toast=True)
                        lp_off += lp_len
                        if debug:
                            yield chunk_id, chunk_seq, chunk_data, (
                                page_bytes, page_header,
                                item_identifiers_bytes, lp_off,
                                item_num, item_identifier, item_header,
                                tupdata)
                        else:
                            yield chunk_id, chunk_seq, chunk_data, None
                else:
                    for i in range(0, len(item_identifiers_bytes),
                                   ITEM_IDENTIFIER_LEN):
                        packed_identifier = item_identifiers_bytes[
                            i:i + ITEM_IDENTIFIER_LEN]
                        tupdata, item_header, item_identifier = \
                            deserialize_item(
                                page_bytes, packed_identifier)
                        dist_key, independent_var_tuple, dependent_var_tuple, \
                            buffer_id = desirialize_tupdata(tupdata)
                        if debug:
                            yield dist_key, \
                                independent_var_tuple, \
                                dependent_var_tuple, \
                                buffer_id, \
                                (page_bytes,
                                 page_header,
                                 item_identifiers_bytes,
                                 None,
                                 item_num,
                                 item_identifier,
                                 item_header,
                                 tupdata)
                        else:
                            yield dist_key, independent_var_tuple, \
                                dependent_var_tuple, buffer_id, None


def table_page_read(page_dir):
    logs("START READING TABLE: {}".format(page_dir))
    page_gen = generator_page(page_dir, debug=False)
    all_data = []
    all_toast = []
    while True:
        try:
            res = next(page_gen)
            dist_key, \
                independent_var_tuple, \
                dependent_var_tuple, \
                buffer_id, \
                _ = res

            all_data.append(res[:-1])
            all_toast.append(
                list(independent_var_tuple[4:]) + [dist_key,
                                                   buffer_id,
                                                   'independent_var'])
            all_toast.append(
                list(dependent_var_tuple[4:]) + [
                    dist_key,
                    buffer_id,
                    'dependent_var'])
        except StopIteration:
            break
    df_data = pd.DataFrame(all_data, columns=[
        'dist_key',
        'independent_var_tuple',
        'dependent_var_tuple',
        'buffer_id'])
    df_toast = pd.DataFrame(all_toast, columns=[
        'va_rawsize',
        'va_extsize',
        'va_valueid',
        'va_toastrelid',
        'external',
        'bytea',
        'dist_key',
        'buffer_id',
        'attname']
    )
    logs("END READING TABLE: {}".format(page_dir))
    return df_data, df_toast


def read_toast_all_bytes(toast_page_path, df_toast, debug=False):
    toast_gen = generator_page(toast_page_path, toast=True, debug=debug)
    grand_bytes = {}
    i = 0
    while True:
        try:
            i += 1
            chunk_id, chunk_seq, chunk_data, debug_data = next(toast_gen)
            if debug:
                (
                    page_bytes,
                    page_header,
                    item_identifiers_bytes,
                    lp_off,
                    item_num,
                    item_identifier,
                    item_header,
                    tupdata
                ) = debug_data
            if int(chunk_id) in df_toast['va_valueid'].values:
                if debug:
                    print(i, chunk_id, chunk_seq)
                if chunk_id not in grand_bytes:
                    grand_bytes[chunk_id] = []
                grand_bytes[chunk_id].append([chunk_seq, chunk_data])
        except StopIteration:
            break
    return grand_bytes


def detoast(grand_bytes, df_toast, df_shape, table_name):
    grand_bytes_con = {}
    df_toast_external = df_toast.loc[df_toast['external'] == True]
    df_toast_internal = df_toast.loc[df_toast['external'] == False]
    # decompress unexternal data:
    for i, row in df_toast_internal.iterrows():
        buffer_id = row['buffer_id']
        bytea = row['bytea']
        attname = row['attname']
        va_rawsize = row['va_rawsize']
        pglz_header = bytea[:8]
        vl_len_, rawsize = struct.unpack('@ii', pglz_header)
        assert rawsize == va_rawsize, 'raw size does not match!'
        dest_data, dest_data_size = pre_alloc_dest(
                bytea, implementation=PGLZ_IMPLEMENTATION)
        dest_data = pglz_decompress(
                    bytea, dest_data, implementation=PGLZ_IMPLEMENTATION)
        shape_info = df_shape.loc[
                    (df_shape['table_name'] == table_name)
                    & (df_shape['buffer_id'] == buffer_id)].iloc[0]
        var_shape = shape_info[attname + '_shape']
        np_data = deserialize_bytea(attname, dest_data, var_shape)
        if buffer_id not in grand_bytes_con:
            grand_bytes_con[buffer_id] = {
                "dependent_var": None, "independent_var": None}
        grand_bytes_con[buffer_id][attname] = np_data
    for k, v in grand_bytes.items():
        logs("ID:{}".format(k))
        va_rawsize, \
            va_extsize, \
            va_valueid, \
            va_toastrelid, \
            external, \
            bytea, \
            dist_key, \
            buffer_id, \
            attname = df_toast_external.loc[
                df_toast_external['va_valueid'] == k].iloc[0]
        shape_info = df_shape.loc[
            (df_shape['table_name'] == table_name)
            & (df_shape['buffer_id'] == buffer_id)].iloc[0]

        var_shape = shape_info[attname + '_shape']
        numchunks = get_num_chunks(va_extsize)
        numchunks_actual = len(v)
        assert numchunks == numchunks_actual

        ressize_include_header = VARHDRSZ + va_extsize
        header_byte = GET_VARSIZE_4B_C(ressize_include_header)
        ressize = va_extsize
        v.sort(key=lambda x: x[0])
        v_parts = [header_byte]
        for idx, chunk in v:
            assert not VARATT_IS_EXTENDED(chunk)
            chunksize = VARSIZE(chunk) - VARHDRSZ
            chunkdata = VARDATA(chunk)
            v_parts.append(chunkdata[:chunksize])
            if ((idx == 0) and (chunksize < ressize)
                    and (chunksize != TOAST_MAX_CHUNK_SIZE)):
                raise Exception("max chunk size error")
            if (idx < numchunks - 1) and (chunksize != TOAST_MAX_CHUNK_SIZE):
                raise Exception("unexpected chunk size")
            if (idx == numchunks - 1) and (
                    (idx * TOAST_MAX_CHUNK_SIZE + chunksize) != ressize):
                raise Exception("unexpected chunk size")

        v = []
        bytea = b''.join(v_parts)
        assert len(bytea) == ressize_include_header, \
            "final size does not match"
        v_parts = []
        dest_data, dest_data_size = pre_alloc_dest(
            bytea, implementation=PGLZ_IMPLEMENTATION)
        logs("START DECOMPRESSING")
        dest_data = pglz_decompress(
            bytea, dest_data, implementation=PGLZ_IMPLEMENTATION)
        logs("END DECOMPRESSING")
        bytea = None
        np_data = deserialize_bytea(attname, dest_data, var_shape)
        dest_data = None
        if buffer_id not in grand_bytes_con:
            grand_bytes_con[buffer_id] = {
                "dependent_var": None, "independent_var": None}

        grand_bytes_con[buffer_id][attname] = np_data
    return grand_bytes_con


def toast_page_read(page_dir, df_toast, df_shape, table_name):
    logs("START READING TOAST: {}".format(page_dir))
    grand_bytes = read_toast_all_bytes(page_dir, df_toast)
    logs("END READING TOAST: {}".format(page_dir))
    logs("START DETOASTING")
    grand_bytes_con = detoast(grand_bytes, df_toast, df_shape, table_name)
    logs("END DETOASTING")
    logs("ITEM NUM:{}, BUFFER_IDs:{}".format(
        len(grand_bytes_con.keys()), grand_bytes_con.keys()))
    return grand_bytes_con


LP_LENGTH_GLOBAL = TOAST_MAX_CHUNK_SIZE + VARHDRSZ + MAX_ALIGN(
    ITEM_HEADER_LEN) + CHUNK_ID_AND_CHUNK_SEQ_LEN
PageHeader = namedtuple('PageHeader', [
    'pd_lsn',
    'pd_tli',
    'pd_flags',
    'pd_lower',
    'pd_upper',
    'pd_special',
    'pd_pagesize_version',
    'pd_prune_xid',
])
HeapTupleHeader = namedtuple('HeapTupleHeader', [
    't_xmin', 't_xmax', 't_cid', 'bi_hi', 'bi_lo', 'ip_posid', 't_infomask2',
    't_infomask', 't_hoff'
])
ItemIdentifier = namedtuple('ItemIdentifier', ['lp_off', 'lp_flags', 'lp_len'])
TupleDesc = namedtuple(
    'TupleDesc',
    ['natts', 'tdtypeid', 'tdtypmod', 'tdrefcount', 'constr', 'attrs'])
FormData_pg_attribute = namedtuple(
    'FormData_pg_attribute',
    ['attrelid', 'attname', 'atttypid', 'attlen', 'attnum', 'attalign'])
Tuple = namedtuple('Tuple', [
    'header',
    'length',
    'dummy1',
    'dummy2',
    'va_rawsize',
    'va_extsize',
    'va_valueid',
    'va_toastrelid',
    'external',
    'bytea'
])
