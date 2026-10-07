import numpy as np
import pandas as pd
from scipy.optimize import minimize
import matplotlib.pyplot as plt

np.random.seed(2026)

assets = ['货币市场基金', '信用债', '股票市场', '房地产信托基金', '对冲基金']
E_R = np.array([5.6, 6.0, 12.2, 11.2, 14.4])
sigma = np.array([13.4, 20.2, 27.6, 24.6, 33.0])
rf = 3.0

cov_matrix = np.array([
    [0.0180, 0.0244, -0.0037, -0.0066, -0.0044],
    [0.0244, 0.0408, -0.0112, -0.0149, -0.0133],
    [-0.0037, -0.0112, 0.0762, 0.0543, 0.0546],
    [-0.0066, -0.0149, 0.0543, 0.0605, 0.0568],
    [-0.0044, -0.0133, 0.0546, 0.0568, 0.1089]
])


# 1. 风险厌恶系数计算
def get_risk_aversion(score):
    return (27 - score) / 18 * 6 + 2


score_zq, score_dqy, score_tm = 18, 24, 12
A_zq_case = 5.15
A_dqy = get_risk_aversion(score_dqy)
A_tm = get_risk_aversion(score_tm)

# 2. 现有组合指标
w_current = np.array([0.1176, 0.1176, 0.3529, 0.1764, 0.2355])
er_current = np.dot(w_current, E_R)
var_current = np.dot(w_current, np.dot(cov_matrix, w_current))
sd_current = np.sqrt(var_current) * 100
sharpe_current = (er_current - rf) / sd_current

# 3. 最优风险资产组合
cov_inv = np.linalg.inv(cov_matrix)
excess_return = (E_R - rf) / 100
w_raw = np.dot(cov_inv, excess_return)
w_opt = w_raw / np.sum(w_raw)

er_opt = np.dot(w_opt, E_R)
var_opt = np.dot(w_opt, np.dot(cov_matrix, w_opt))
sd_opt = np.sqrt(var_opt) * 100
sharpe_opt = (er_opt - rf) / sd_opt


# 4. 最优资本配置比例y*
def get_capital_allocation(A, er_p, sd_p, r_free):
    return (er_p - r_free) / (0.01 * A * (sd_p ** 2))


y_zq = get_capital_allocation(A_zq_case, er_opt, sd_opt, rf)
y_dqy = get_capital_allocation(A_dqy, er_opt, sd_opt, rf)
y_tm = get_capital_allocation(A_tm, er_opt, sd_opt, rf)

er_c_zq = rf + y_zq * (er_opt - rf)
sd_c_zq = y_zq * sd_opt


# 5. 15年现金流仿真
def simulate_cash_flows(rate, liq_rate=0.0560, unemployment=None):
    w_inv = 35.0
    w_liq = 15.0

    history = []

    for yr in range(1, 16):

        salary = 20.0 if yr <= 5 else (
            25.0 if yr <= 10 else 30.0
        )

        # 失业
        if unemployment and unemployment[0] <= yr <= unemployment[1]:
            salary *= (1 - unemployment[2])

        # 租金收入
        rent = 3.0 if yr >= 6 else 0.0

        # 家庭支出
        living = 10.0
        mortgage = 8.0 if 6 <= yr <= 15 else 0.0

        # 大额支出
        lump_sum = 30.0 if yr == 5 else (
            25.0 if yr == 10 else 0.0
        )

        # 当年净现金流
        net_cf = (
            salary
            + rent
            - living
            - mortgage
            - lump_sum
        )

        # 年初余额
        start_inv = w_inv
        start_liq = w_liq

        # 投资资金收益
        inv_gain = start_inv * rate

        # 流动性资金收益
        liq_gain = start_liq * liq_rate

        # 年末余额
        w_inv = start_inv + inv_gain + net_cf
        w_liq = start_liq + liq_gain

        # 年末金融资产
        total_wealth = w_inv + w_liq

        history.append({
            'Year': yr,
            'Income': salary + rent,
            'Living': living,
            'Mortgage': mortgage,
            'LumpSum': lump_sum,
            'NetCF': net_cf,
            'InvStart': start_inv,
            'InvReturn': inv_gain,
            'InvEnd': w_inv,
            'LiqEnd': w_liq,
            'FinancialAssets': total_wealth
        })

    return pd.DataFrame(history)


# 基准情景
df_base = simulate_cash_flows(
    er_c_zq / 100
)

# 乐观情景
df_opt_case = simulate_cash_flows(
    (er_c_zq + sd_c_zq) / 100
)

# 悲观情景
df_pess_case = simulate_cash_flows(
    (er_c_zq - sd_c_zq) / 100,
    liq_rate=0.0300,
    unemployment=(7, 8, 0.40)
)

# 6. 蒙特卡罗模拟
def monte_carlo(n_sims=10000):
    mc_totals = []

    for _ in range(n_sims):
        w_inv, w_liq = 35.0, 15.0
        rets = np.random.normal(0.0710, 0.0893, 15)

        for yr in range(1, 16):
            salary = 20.0 if yr <= 5 else (
                25.0 if yr <= 10 else 30.0
            )

            rent = 3.0 if yr >= 6 else 0.0
            living = 10.0
            mortgage = 8.0 if 6 <= yr <= 15 else 0.0

            lump = 30.0 if yr == 5 else (
                25.0 if yr == 10 else 0.0
            )

            net_cf = (
                salary + rent
                - living
                - mortgage
                - lump
            )

            w_inv = w_inv * (1 + rets[yr - 1]) + net_cf
            w_liq = w_liq * (1 + 0.0560)

        mc_totals.append(w_inv + w_liq)

    return np.array(mc_totals)

mc_results = monte_carlo()
percentiles = np.percentile(mc_results, [5, 25, 50, 75, 95])

# 蒙特卡罗模拟结果分布图
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

plt.figure(figsize=(8, 5))

plt.hist(
    mc_results,
    bins=35,
    edgecolor='black',
    linewidth=0.5,
    alpha=0.75
)

median = percentiles[2]

plt.axvline(
    median,
    linestyle='--',
    linewidth=1.8
)

plt.text(
    median + 5,
    plt.ylim()[1] * 0.88,
    f'50%分位数：{median:.2f}万元',
    fontsize=10
)

plt.xlabel('第15年末金融资产（万元）')
plt.ylabel('模拟次数')

plt.tight_layout()
plt.show()

if __name__ == '__main__':
    print(f"张强现有组合: E(R)={er_current:.2f}%, sigma={sd_current:.2f}%, Sharpe={sharpe_current:.4f}")
    print(f"最优切点组合: E(R)={er_opt:.2f}%, sigma={sd_opt:.2f}%, Sharpe={sharpe_opt:.4f}")
    print(f"切点各资产相对比例: {np.round(w_raw, 5)}")
    print(f"切点权重: {np.round(w_opt * 100, 2)}")
    print(f"资本配置比例 y*: 张强={y_zq * 100:.2f}%, 邓权艺={y_dqy * 100:.2f}%, 童萌={y_tm * 100:.2f}%")
    print(f"15年末金融资产 (基准) = {df_base.iloc[-1]['FinancialAssets']:.2f} 万元")
    print(f"15年末金融资产 (乐观) = {df_opt_case.iloc[-1]['FinancialAssets']:.2f} 万元")
    print(f"15年末金融资产 (悲观) = {df_pess_case.iloc[-1]['FinancialAssets']:.2f} 万元")
    print("===== 基准情景 =====")
    print(df_base)

    print("\n===== 乐观情景 =====")
    print(df_opt_case)

    print("\n===== 悲观情景 =====")
    print(df_pess_case)
    print(f"蒙特卡罗 5% 分位数 = {percentiles[0]:.2f} 万元")
    print(f"蒙特卡罗 25% 分位数 = {percentiles[1]:.2f} 万元")
    print(f"蒙特卡罗 50% 分位数 = {percentiles[2]:.2f} 万元")
    print(f"蒙特卡罗 75% 分位数 = {percentiles[3]:.2f} 万元")
    print(f"蒙特卡罗 95% 分位数 = {percentiles[4]:.2f} 万元")

