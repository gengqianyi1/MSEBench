# Task 05 公开数据

`inputs/` 包含：

- `monomer_A.xyz`：咖啡因（PubChem CID 2519）；
- `monomer_B.xyz`：水（PubChem CID 962）；
- `complex_pose_01.xyz`–`complex_pose_14.xyz`：14 个匿名刚体起始构型；
- `monomer_manifest.csv`、`complex_manifest.csv`；
- `fragment_manifest.csv`：每个复合物逐原子的 A/B 归属、片段内编号和元素；
- `SOURCE.md` 与 `public_data_manifest.json`。

候选文件名和 XYZ 注释不公开初始接触类别、优化后坍缩关系或能量排序。模型必须依据优化后真实几何和 xTB 证据完成接触重建与聚类。

所有初始复合物满足最短跨片段原子距离不小于 1.20 Å、最短跨片段重原子距离不小于 2.20 Å。复合物原子编号连续覆盖全部原子，每个原子恰好属于 A 或 B，两个片段均非空。

公开数据不包含结合能、相互作用能、形变能、参考排名或任何 Golden 数值。
