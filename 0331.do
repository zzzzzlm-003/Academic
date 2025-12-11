cd E:\备份资料\毕业论文\数据处理\模型一变量构造\结果
* 描述性统计（所有ABR变量及模型其他变量）
summarize ABR_* Deviation I multi single star dmy institute, detail

* 使用tabstat生成更详细清晰的描述统计表
tabstat ABR_* Deviation I multi single star dmy institute, ///
        statistics(mean sd median min max) columns(statistics)

esttab using "Descriptive_Statistics.rtf", ///
cells("mean sd min p50 max") ///
label ///
title("Descriptive Statistics for All ABRs and Model Variables") ///
replace

* 计算并导出相关系数矩阵
estpost corr ABR_* Deviation I multi single star dmy institute, matrix listwise

esttab using "相关系数矩阵.rtf", ///
unstack not noobs compress ///
star(* 0.05 ** 0.01 *** 0.001) ///
title("Correlation Matrix for ABRs and Model Variables") replace
