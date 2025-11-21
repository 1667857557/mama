#!/usr/bin/env python3

"""
MAMA LD score generation — 流式 / 低峰值内存实现（使用 SQLite 做磁盘合并）
"""

import argparse as argp
import contextlib
import gc
import io
from itertools import combinations
import logging
import os
import sys
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import sqlite3
import tempfile

from mama import (numpy_err_handler, output_prefix, setup_func)
from pop_info import PopInfo
from util.bed import BED_SUFFIX
from util.bim import BIM_BP_COL, BIM_CM_COL, BIM_COL_TYPES, BIM_SUFFIX
from util.fam import FAM_SUFFIX
from util.sumstats import SNP_COL

__version__ = '1.0.0'
INPUT_SEP = ","
SOFTWARE_CORRESPONDENCE_EMAIL = "jjala.ssgac@gmail.com"
OTHER_CORRESPONDENCE_EMAIL = "paturley@broadinstitute.org"

HEADER = f"""
MAMA LD SCORE GENERATION: Version {__version__}
Software: {SOFTWARE_CORRESPONDENCE_EMAIL}
"""

DEFAULT_SHORT_PREFIX = "mama_ldscore"
DEFAULT_FULL_OUT_PREFIX = os.path.join(os.getcwd(), DEFAULT_SHORT_PREFIX)
DEFAULT_BP_THRESHOLD = BIM_COL_TYPES[BIM_BP_COL](10**6)
DEFAULT_OUT_SEP = "\t"
DEFAULT_OUT_NAN = "NaN"

POPS = 'pop_tuple'
NUM_POPS = 'num_pops'
GENDATA_PATHS = 'gendata_paths'
STANDARDIZE = 'standardize'
WINDOW_COL = 'window_col'
WINDOW_THRESHOLD = 'window_threshold'
OUT_PREFIX = 'out_prefix'


def non_negative(numeric_type):
    def check_non_negative(val_to_be_checked: numeric_type):
        number_as_type = numeric_type(val_to_be_checked)
        if number_as_type < 0:
            raise argp.ArgumentTypeError("Value specified (%s) must be >= 0" % number_as_type)
        return number_as_type
    return check_non_negative


def bedbimfam_prefix(prefix: str):
    prefix = prefix.strip()
    for suffix in (BED_SUFFIX, BIM_SUFFIX, FAM_SUFFIX):
        if not os.path.exists(prefix + suffix):
            raise argp.ArgumentTypeError("The input file [%s%s] does not exist." %
                                         (prefix, suffix))
    return prefix


def gendata_pair(s_input: str) -> Tuple[str, str]:
    try:
        file_prefix, ancestry = map(lambda x: x.strip(), s_input.split(INPUT_SEP))
    except Exception as exc:
        raise argp.ArgumentTypeError("Error parsing %s into file prefix and ancestry" %
                                     s_input) from exc
    return ancestry, bedbimfam_prefix(file_prefix)


def get_ldscore_parser(progname: str) -> argp.ArgumentParser:
    parser = argp.ArgumentParser(prog=progname)
    in_opt = parser.add_argument_group(title="Main Input Specifications")
    in_opt.add_argument("--gendata", type=gendata_pair, nargs="+", required=True,
                        metavar="FILE_PREFIX%sANCESTRY" % INPUT_SEP,
                        help="List of pairs FP%sA where FP is path prefix to a set of bed/bim/fam"
                             "files and A is the name of an ancestry.  The ancestry is used for "
                             "to generate LD scores labeled by ANC1_ANC2." % INPUT_SEP)
    out_opt = parser.add_argument_group(title="Output Specifications")
    out_opt.add_argument("--out", metavar="FILE_PREFIX", type=output_prefix,
                         default=DEFAULT_FULL_OUT_PREFIX,
                         help="Full prefix of output files (logs, LD score file, etc.).")
    gen_opt = parser.add_argument_group(title="General Options")
    gen_opt.add_argument("--use-standardized-units", default=False, action="store_true")
    log_opt = gen_opt.add_mutually_exclusive_group()
    log_opt.add_argument("--quiet", action="store_true")
    log_opt.add_argument("--verbose", action="store_true")
    reg_opt = parser.add_argument_group(title="Window Specifications")
    reg_ld_opt = reg_opt.add_mutually_exclusive_group()
    reg_ld_opt.add_argument("--window-bp", type=non_negative(BIM_COL_TYPES[BIM_BP_COL]),
                            metavar="NUM_BASEPAIRS")
    reg_ld_opt.add_argument("--window-cm", type=non_negative(BIM_COL_TYPES[BIM_CM_COL]),
                            metavar="NUM_CM")
    return parser


def validate_inputs(pargs: argp.Namespace, user_args: Dict[str, Any]):
    logging.debug("\nProgram was called with the following arguments:\n%s", user_args)
    internal_values = dict()
    internal_values[GENDATA_PATHS] = dict(pargs.gendata)
    internal_values[POPS] = tuple(internal_values[GENDATA_PATHS].keys())
    internal_values[NUM_POPS] = len(internal_values[POPS])
    internal_values[STANDARDIZE] = pargs.use_standardized_units
    if pargs.window_bp:
        internal_values[WINDOW_COL] = BIM_BP_COL
        internal_values[WINDOW_THRESHOLD] = pargs.window_bp
    elif pargs.window_cm:
        internal_values[WINDOW_COL] = BIM_CM_COL
        internal_values[WINDOW_THRESHOLD] = pargs.window_cm
    else:
        internal_values[WINDOW_COL] = BIM_BP_COL
        internal_values[WINDOW_THRESHOLD] = DEFAULT_BP_THRESHOLD
    internal_values[OUT_PREFIX] = pargs.out
    return internal_values


def _ensure_dataframe_like(obj, colname_hint: str = None) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame):
        df = obj
        if df.shape[1] == 1 and (df.columns is None or df.columns.tolist() == [0]):
            if colname_hint is not None:
                df.columns = [colname_hint]
        return df
    if isinstance(obj, pd.Series):
        name = obj.name if obj.name is not None else colname_hint
        return obj.to_frame(name=name)
    if isinstance(obj, np.ndarray):
        if obj.ndim == 1:
            df = pd.DataFrame(obj)
        else:
            df = pd.DataFrame(obj)
        if df.shape[1] == 1 and colname_hint is not None:
            df.columns = [colname_hint]
        return df
    return pd.DataFrame(obj)


def _sanitize_colname(s: str) -> str:
    # 尽量保证列名能作为 SQLite 列名：用下划线替代空白，去掉/替换危险字符
    return "".join(c if c.isalnum() or c == "_" else "_" for c in str(s))


def main_func(argv: List[str]):
    parsed_args, user_args = setup_func(argv, get_ldscore_parser, HEADER)

    np.seterr(all='call')
    np.seterrcall(numpy_err_handler)

    if logging.root.level <= logging.DEBUG:
        logging.debug("\nPrinting Pandas' version summary:")
        with contextlib.redirect_stdout(io.StringIO()) as f:
            pd.show_versions()
        logging.debug("%s\n", f.getvalue())

    try:
        iargs = validate_inputs(parsed_args, user_args)

        logging.info("\nPerforming per-population calculations (filtering and correlations)...")
        popinfo = {p: PopInfo(pop_id=p, bedbimfam_prefix=iargs[GENDATA_PATHS][p],
                              dist_col=iargs[WINDOW_COL], win_size=iargs[WINDOW_THRESHOLD],
                              standardize=iargs[STANDARDIZE],
                              r_band_filename="%s_%s_R.npy" % (iargs[OUT_PREFIX], p))
                   for p in iargs[POPS]}

        logging.info("\nCalculating cross-population indices...")
        for p1info, p2info in combinations(popinfo.values(), 2):
            p1info.calc_cross_pop_indices(p2info)

        # 如果 PopInfo 有释放中间缓存的方法，尽早调用
        for p in popinfo.values():
            if hasattr(p, 'free_intermediate_buffers') and callable(getattr(p, 'free_intermediate_buffers')):
                try:
                    p.free_intermediate_buffers()
                except Exception:
                    logging.debug("free_intermediate_buffers failed for pop %s", getattr(p, 'pop_id', '<unknown>'))

        # 排序以优化内存/IO（与之前逻辑一致）
        sorted_pop_tuple = sorted((pop_obj for pop_obj in popinfo.values()),
                                  key=lambda p: p.M * p.max_lower_extent, reverse=True)

        # popinfo 已不再需要，释放引用
        del popinfo
        gc.collect()

        logging.info("\nCalculating LD scores and merging on disk (SQLite)...")

        # 准备 SQLite DB（在输出前缀目录下）
        db_path = f"{iargs[OUT_PREFIX]}_ldscores.db"
        if os.path.exists(db_path):
            os.remove(db_path)
        conn = sqlite3.connect(db_path, timeout=60)
        cur = conn.cursor()
        # 加速写入的 PRAGMA
        cur.execute("PRAGMA synchronous = OFF;")
        cur.execute("PRAGMA journal_mode = MEMORY;")
        cur.execute("PRAGMA temp_store = MEMORY;")
        conn.commit()

        # 创建主表（仅含 SNP 主键），其余列后续 ALTER ADD
        cur.execute('CREATE TABLE IF NOT EXISTS ldscores (snp TEXT PRIMARY KEY);')
        conn.commit()

        n_pops = len(sorted_pop_tuple)
        for p1_index in range(n_pops):
            p1_info = sorted_pop_tuple[p1_index]
            p1_r = p1_info.get_banded_R()

            for p2_index in range(p1_index, n_pops):
                p2_info = sorted_pop_tuple[p2_index]

                # 计算
                col_obj = p1_info.calc_ldscores(p2_info, self_mat=p1_r)

                # 转 DataFrame
                try:
                    id1 = getattr(p1_info, 'id', getattr(p1_info, 'pop_id', f'P{p1_index}'))
                except Exception:
                    id1 = f'P{p1_index}'
                try:
                    id2 = getattr(p2_info, 'id', getattr(p2_info, 'pop_id', f'P{p2_index}'))
                except Exception:
                    id2 = f'P{p2_index}'
                colname_hint = f"{id1}_{id2}"

                col_df = _ensure_dataframe_like(col_obj, colname_hint=colname_hint)

                # 确保 index 都是字符串形式的 SNP id
                col_df.index = col_df.index.map(str)
                # 若没有 index name，设为 SNP_COL（便于后续）
                if col_df.index.name is None:
                    col_df.index.name = SNP_COL

                # 对每一列进行 upsert 到 SQLite
                for col_idx, original_col in enumerate(col_df.columns):
                    col_series = col_df.iloc[:, col_idx]
                    colname = _sanitize_colname(original_col or f"{colname_hint}_{col_idx}")

                    # 动态添加列（若已存在会抛异常）
                    try:
                        cur.execute(f'ALTER TABLE ldscores ADD COLUMN "{colname}"')
                    except sqlite3.OperationalError:
                        # 列已存在或其他问题；继续（列已存在是正常情况）
                        pass

                    # 创建临时表并批量插入值（使用事务）
                    cur.execute('CREATE TEMP TABLE tmp (snp TEXT PRIMARY KEY, val);')
                    # 转换 NaN -> None
                    rows = ((snp, None if (pd.isna(val)) else float(val)) for snp, val in zip(col_series.index.tolist(), col_series.values.tolist()))
                    cur.executemany('INSERT INTO tmp (snp, val) VALUES (?, ?);', rows)

                    # 将新的 SNP 插入主表（若不存在）
                    cur.execute('INSERT OR IGNORE INTO ldscores(snp) SELECT snp FROM tmp;')
                    # 更新对应列的值（存在则覆盖）
                    cur.execute(f'UPDATE ldscores SET "{colname}" = (SELECT val FROM tmp WHERE tmp.snp = ldscores.snp) WHERE EXISTS (SELECT 1 FROM tmp WHERE tmp.snp = ldscores.snp);')

                    # 删除临时表
                    cur.execute('DROP TABLE tmp;')
                    conn.commit()

                # 释放临时 Python 对象并触发 GC
                del col_obj, col_df
                gc.collect()

            del p1_r
            # 断开引用，帮助释放
            sorted_pop_tuple[p1_index] = None
            gc.collect()

        # 所有列都已写入 SQLite，现在导出为最终 CSV（分块读取）
        out_txt = f"{iargs[OUT_PREFIX]}_ldscores.txt"
        # 使用 pandas 的 read_sql_query 且 chunksize 分块写出
        query = 'SELECT * FROM ldscores ORDER BY snp;'
        chunksize = 100000
        first_chunk = True
        for chunk in pd.read_sql_query(query, conn, index_col='snp', chunksize=chunksize):
            # pandas 会把 None 转为 NaN，写出时用 na_rep
            chunk.to_csv(out_txt, sep=DEFAULT_OUT_SEP, na_rep=DEFAULT_OUT_NAN, mode='w' if first_chunk else 'a', header=first_chunk, index=True)
            first_chunk = False
            gc.collect()

        conn.close()
        logging.info("\nExecution complete. Final LD score file: %s\n", out_txt)

    except Exception as exc:  # pylint: disable=broad-except
        logging.exception(exc)
        sys.exit(1)


if __name__ == '__main__':
    main_func(sys.argv)
