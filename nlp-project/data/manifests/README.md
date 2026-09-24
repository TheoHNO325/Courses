# 数据登记表说明

`datasets.csv` 是项目的数据来源总表。任何文件下载前先登记；下载后补充日期、哈希、实际条数和本地相对路径。

字段说明：

- `dataset_id`：稳定且唯一的内部标识；
- `name`、`version`：公开名称和版本；
- `task`：translation、evaluation、style_analysis、ocr、detection 或 multimodal；
- `language_pair`：语言方向或模态；
- `domain`：领域；
- `source_style`：当前推定的源文风格；
- `data_origin`：人工翻译、机器生成、人工标注或未知；
- `source_url`：官方入口；
- `license`：许可证；不明确时使用 `TO_VERIFY`；
- `access`：public、gated、sample_only 等；
- `allowed_use`：计划用途；
- `local_path`：下载后的项目内相对路径；
- `download_date`：下载日期；
- `sha256`：原始文件哈希；
- `record_count`：已核验的记录数；
- `split_policy`：数据划分要求；
- `status`：planned、deferred、application_pending、downloaded、verified 等；
- `notes`：风险或待确认事项。

注意：表中的初始许可证和条数来自项目调研，下载时仍需对官方页面和实际文件再次核验。
