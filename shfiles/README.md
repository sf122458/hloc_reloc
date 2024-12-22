### 开始任务

开始构建数据集：
```shell
sh start_db.sh
```

开始重定位任务：
```shell
sh start_loc.sh
```


### 构建数据集
设定坐标：
```shell
sh set_pos.sh x y z
```

拍照以及记录无人机位置：
```shell
sh takephoto.sh
```


### 重定位
设定坐标：
```shell
sh set_pos.sh x y z
```

添加矫正图片：
```shell
sh add_cali.sh
```

计算矫正矩阵：
```shell
sh calc_cali.sh
```

```shell
sh reloc.sh [Path]
```