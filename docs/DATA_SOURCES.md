# 数据来源与准备

本发布包不携带商业个股数据、作者因子原文件、论文全文或浏览器下载地址。没有配置远程 Git 仓库，因此本次未进行远程分支合并或同步。

## CSMAR

自行授权导出的六类文件、路径和字段说明见根目录 README 与学习手册。保持 CSV 原始列名、证券代码前导零、日期与计量单位；真实复现脚本读取 `private_data/csmar_raw/`。不要把数据放在公开 results 目录。

## 作者公开资料

- [Stambaugh 作者主页及数据入口](https://finance.wharton.upenn.edu/~stambaug/)
- [Online Appendix](https://finance.wharton.upenn.edu/~stambaug/size_value_china_appendix_2_rev.pdf)

本地历史对照使用 `CH3_factors_monthly_202608.xlsx`，标准化目标是 `author_data/verified_20260925/ch3_monthly_official.csv`；列 `date,RF_MONTHLY,MKT,SMB,VMG`，收益采用小数、日期为月末。文件需覆盖 2008—2025 年。后续版本可能修订历史数值。

`scripts/prepare_ch3_public_data.py` 是原研究的多文件标准化脚本，依赖 CH-3 工作簿、六组合工作簿、CH-4 交叉核对文件及来源归档；它不自动下载数据。详见脚本中 DATA 路径与输入文件名。只有已标准化官方 CSV 也可直接运行 CSMAR 实证入口。

## 无商业数据

合成数据由固定种子生成，运行 `bash scripts/ch3_pipeline.sh smoke-v2` 即可生成并检验。它验证工程行为，不代表市场业绩。原始数据、生成数据与运行明细均由 .gitignore 排除。

## 汇总结果

`results/20260926/` 只包含模型层面的统计表、固定规格和验证记录；`assets/` 只包含汇总图。未分发逐股面板、持仓、逐股排除表或作者原序列。其结果口径与局限见 README。
