#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CHIP 2018 真实数据分析 - 使用验证过的变量
绝对不使用任何估计值
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
import warnings
warnings.filterwarnings('ignore')

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def analyze_chip_real_data():
    """分析CHIP 2018真实数据"""
    print("="*80)
    print("CHIP 2018 真实数据分析")
    print("="*80)
    
    # 1. 加载数据
    print("\n1. 加载真实数据...")
    urban_data = pd.read_stata('../data/chip2018_urban_p.dta', convert_categoricals=False)
    rural_data = pd.read_stata('../data/chip2018_rural_p.dta', convert_categoricals=False)
    
    print(f"   城镇原始样本: {len(urban_data):,}")
    print(f"   农村原始样本: {len(rural_data):,}")
    print(f"   总原始样本: {len(urban_data) + len(rural_data):,}")
    
    # 2. 数据预处理
    print("\n2. 数据预处理...")
    
    # 添加城乡标识
    urban_data['urban'] = 1
    rural_data['urban'] = 0
    
    # 合并数据
    data = pd.concat([urban_data, rural_data], ignore_index=True)
    print(f"   合并后样本: {len(data):,}")
    
    # 3. 变量处理
    print("\n3. 处理关键变量...")
    
    # 计算年龄
    data['age'] = 2018 - data['A04_1']
    
    # 性别（男=1）
    data['male'] = (data['A03'] == 1).astype(int)
    
    # 教育年限
    data['edu_years'] = data['A13_3']
    
    # 工作经验
    data['experience'] = data['age'] - data['edu_years'] - 6
    data['experience'] = data['experience'].clip(0, 50)
    data['experience_sq'] = data['experience'] ** 2
    
    # 收入
    data['income'] = data['C05_1']
    
    # 工作月数
    data['work_months'] = data['C01_1']
    
    # 4. 数据筛选
    print("\n4. 数据筛选...")
    
    # 筛选前统计
    print(f"   筛选前样本: {len(data):,}")
    
    # 年龄限制（25-60岁）
    data = data[(data['age'] >= 25) & (data['age'] <= 60)]
    print(f"   年龄25-60岁: {len(data):,}")
    
    # 正收入
    positive_income_count = (data['income'] > 0).sum()
    print(f"   正收入样本: {positive_income_count:,} ({positive_income_count/len(data)*100:.1f}%)")
    
    # 筛选正收入
    data = data[data['income'] > 0]
    
    # 工作时间合理（至少工作3个月）
    data = data[data['work_months'] >= 3]
    print(f"   工作≥3个月: {len(data):,}")
    
    # 教育年限合理（0-22年）
    data = data[(data['edu_years'] >= 0) & (data['edu_years'] <= 22)]
    print(f"   教育年限0-22: {len(data):,}")
    
    # 删除缺失值
    before_dropna = len(data)
    data = data.dropna(subset=['income', 'edu_years', 'age', 'male', 'experience'])
    print(f"   删除缺失值: {before_dropna:,} -> {len(data):,}")
    
    # 创建对数收入
    data['log_income'] = np.log(data['income'])
    
    print(f"\n   最终分析样本: {len(data):,}")
    print(f"   样本保留率: {len(data)/(len(urban_data)+len(rural_data))*100:.1f}%")
    
    # 5. 描述性统计
    print("\n5. 描述性统计...")
    print("-"*50)
    
    desc_vars = ['income', 'edu_years', 'age', 'male', 'urban', 'work_months']
    desc_stats = data[desc_vars].describe()
    
    print("\n主要变量统计:")
    print(f"   年收入: 均值={data['income'].mean():,.0f}, 中位数={data['income'].median():,.0f}")
    print(f"   教育年限: 均值={data['edu_years'].mean():.1f}, 标准差={data['edu_years'].std():.1f}")
    print(f"   年龄: 均值={data['age'].mean():.1f}, 标准差={data['age'].std():.1f}")
    print(f"   男性比例: {data['male'].mean()*100:.1f}%")
    print(f"   城镇比例: {data['urban'].mean()*100:.1f}%")
    print(f"   平均工作月数: {data['work_months'].mean():.1f}")
    
    # 6. 明瑟方程回归
    print("\n6. 明瑟方程回归...")
    print("-"*50)
    
    # 基本回归
    formula = 'log_income ~ edu_years + experience + experience_sq + male + urban'
    model = ols(formula, data=data).fit()
    
    print("\n全样本回归结果:")
    print(f"   样本量: {len(data):,}")
    print(f"   教育回报率: {model.params['edu_years']*100:.3f}%")
    print(f"   标准误: {model.bse['edu_years']*100:.3f}%")
    print(f"   t统计量: {model.tvalues['edu_years']:.3f}")
    print(f"   p值: {model.pvalues['edu_years']:.6f}")
    print(f"   95%置信区间: [{model.conf_int().loc['edu_years', 0]*100:.3f}%, {model.conf_int().loc['edu_years', 1]*100:.3f}%]")
    print(f"   R²: {model.rsquared:.4f}")
    print(f"   调整R²: {model.rsquared_adj:.4f}")
    print(f"   F统计量: {model.fvalue:.2f} (p={model.f_pvalue:.6f})")
    
    # 分城乡回归
    print("\n分城乡回归:")
    
    # 城镇
    urban_df = data[data['urban'] == 1]
    if len(urban_df) > 100:
        formula_sub = 'log_income ~ edu_years + experience + experience_sq + male'
        model_urban = ols(formula_sub, data=urban_df).fit()
        print(f"\n城镇样本 (n={len(urban_df):,}):")
        print(f"   教育回报率: {model_urban.params['edu_years']*100:.3f}%")
        print(f"   标准误: {model_urban.bse['edu_years']*100:.3f}%")
        print(f"   p值: {model_urban.pvalues['edu_years']:.6f}")
        print(f"   R²: {model_urban.rsquared:.4f}")
    
    # 农村
    rural_df = data[data['urban'] == 0]
    if len(rural_df) > 100:
        model_rural = ols(formula_sub, data=rural_df).fit()
        print(f"\n农村样本 (n={len(rural_df):,}):")
        print(f"   教育回报率: {model_rural.params['edu_years']*100:.3f}%")
        print(f"   标准误: {model_rural.bse['edu_years']*100:.3f}%")
        print(f"   p值: {model_rural.pvalues['edu_years']:.6f}")
        print(f"   R²: {model_rural.rsquared:.4f}")
    
    # 7. 数据质量诊断
    print("\n7. 数据质量诊断...")
    print("-"*50)
    
    # 收入分布
    print("\n收入分布:")
    income_percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    for p in income_percentiles:
        value = data['income'].quantile(p/100)
        print(f"   {p:2d}%分位: {value:,.0f}")
    
    # 检查异常值
    print("\n异常值检查:")
    print(f"   收入=0: {(data['income'] == 0).sum()}")
    print(f"   收入<1000: {(data['income'] < 1000).sum()}")
    print(f"   收入>1000000: {(data['income'] > 1000000).sum()}")
    print(f"   教育年限<0: {(data['edu_years'] < 0).sum()}")
    print(f"   教育年限>22: {(data['edu_years'] > 22).sum()}")
    
    # 8. 生成汇总表
    print("\n8. 生成汇总表...")
    print("="*80)
    
    summary_dict = {
        '指标': [
            '原始样本量',
            '最终样本量', 
            '样本保留率(%)',
            '平均年收入(元)',
            '收入中位数(元)',
            '平均教育年限',
            '平均年龄',
            '男性比例(%)',
            '城镇比例(%)',
            '教育回报率(%)',
            '标准误(%)',
            'p值',
            'R²'
        ],
        '数值': [
            len(urban_data) + len(rural_data),
            len(data),
            len(data)/(len(urban_data)+len(rural_data))*100,
            data['income'].mean(),
            data['income'].median(),
            data['edu_years'].mean(),
            data['age'].mean(),
            data['male'].mean()*100,
            data['urban'].mean()*100,
            model.params['edu_years']*100,
            model.bse['edu_years']*100,
            model.pvalues['edu_years'],
            model.rsquared
        ]
    }
    
    summary_df = pd.DataFrame(summary_dict)
    
    print("\n汇总结果表:")
    for _, row in summary_df.iterrows():
        if row['指标'] in ['原始样本量', '最终样本量']:
            print(f"{row['指标']:20s}: {row['数值']:,.0f}")
        elif row['指标'] in ['平均年收入(元)', '收入中位数(元)']:
            print(f"{row['指标']:20s}: {row['数值']:,.0f}")
        elif row['指标'] == 'p值':
            print(f"{row['指标']:20s}: {row['数值']:.6f}")
        else:
            print(f"{row['指标']:20s}: {row['数值']:.3f}")
    
    # 保存结果
    results = {
        'data': data,
        'model': model,
        'summary': summary_df,
        'stats': {
            'total_sample': len(urban_data) + len(rural_data),
            'final_sample': len(data),
            'retention_rate': len(data)/(len(urban_data)+len(rural_data))*100,
            'positive_income_rate': positive_income_count/(len(urban_data)+len(rural_data))*100,
            'mean_income': data['income'].mean(),
            'median_income': data['income'].median(),
            'mean_edu': data['edu_years'].mean(),
            'return_rate': model.params['edu_years']*100,
            'return_se': model.bse['edu_years']*100,
            'return_pvalue': model.pvalues['edu_years'],
            'r_squared': model.rsquared
        }
    }
    
    print("\n" + "="*80)
    print("✓ CHIP 2018真实数据分析完成！")
    print("="*80)
    
    return results

if __name__ == "__main__":
    results = analyze_chip_real_data()