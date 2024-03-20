import numpy as np
import pandas as pd

def woe(good_pct, bad_pct):
    return np.log(good_pct / bad_pct)

def iv(good_pct, bad_pct):
    return (good_pct - bad_pct) * np.log(good_pct / bad_pct)

def calculate_iv_with_seg_cols(self, seg_cols):
    if len(seg_cols) < len(self.all_seg_cols):
        drop_cols = list(set(self.all_seg_cols).difference(seg_cols))
        factor_good = self.factor_g_df.drop(columns=drop_cols)
        factor_bad = self.factor_b_df.drop(columns=drop_cols)
    else:
        factor_good = self.factor_g_df
        factor_bad = self.factor_b_df

    factor_good = factor_good.groupby(seg_cols, dropna=False).sum()
    factor_bad = factor_bad.groupby(seg_cols, dropna=False).sum()

    factor_good_sum = np.add.reduceat(factor_good.values, indices=self.ranges[:-1], axis=1)
    factor_good_sum = np.repeat(factor_good_sum, self.num_cats_per_col, axis=1)
    factor_bad_sum = np.add.reduceat(factor_bad.values, indices=self.ranges[:-1], axis=1)
    factor_bad_sum = np.repeat(factor_bad_sum, self.num_cats_per_col, axis=1)

    factor_good_pct = factor_good / factor_good_sum
    factor_bad_pct = factor_bad / factor_bad_sum

    woe_val = woe(factor_good_pct, factor_bad_pct)
    iv_val = iv(factor_good_pct, factor_bad_pct)

    woe_df = pd.DataFrame(woe_val, index=factor_good_pct.index, columns=self.bin_info['bin_idx'])
    woe_df = woe_df.stack().reset_index()
    column_names = woe_df.columns.tolist()
    column_names[-2:] = ['bin_idx', "WoE"]
    woe_df.columns = column_names

    iv_col = iv_val.values.flatten()
    good_factor_cnt = factor_good.values.flatten()
    bad_factor_cnt = factor_bad.values.flatten()
    total_factor_cnt = good_factor_cnt + bad_factor_cnt
    good_pct = factor_good_pct.values.flatten()
    bad_pct = factor_bad_pct.values.flatten()
    pct = (good_factor_cnt + bad_factor_cnt) / (factor_good_sum + factor_bad_sum)

    columns = ['good_factor_cnt', 'bad_factor_cnt', 'total_factor_cnt', "good_pct", "bad_pct", "pct", "IV"]
    woe_df[columns] = np.stack([good_factor_cnt, bad_factor_cnt, total_factor_cnt, 
                                good_pct, bad_pct, pct, iv_col]).T

    woe_df['bin'] = np.tile(self.bin_info['bin'], factor_good_pct.shape[0])
    woe_df['var'] = np.tile(self.bin_info['var'], factor_good_pct.shape[0])

    iv_val = np.add.reduceat(iv_val.values, indices=self.ranges[:-1], axis=1)
    iv_df = pd.DataFrame(iv_val, index=factor_good_pct.index, columns=self.var_names)
    iv_df = iv_df.stack().reset_index()
    column_names = iv_df.columns.tolist()
    column_names[-2:] = ['var', 'IV']
    iv_df.columns = column_names

    return woe_df, iv_df
