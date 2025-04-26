# PyCharm 连接免费英伟达 4090D GPU 服务器实战（本文提供项目代码、英伟达4090D显卡服务器完整环境）

-   pycharm 专业版
    -   使用本地编译器 实时将本地代码同步至远程 GPU 工作站(推荐)
-   pycharm 社区版
    -   使用本地编译器 手动将本地代码同步至远程 GPU 工作站

<!-- TOC -->

-   [PyCharm 连接免费英伟达 4090D GPU 服务器实战](#PyCharm 连接免费英伟达 4090D GPU 服务器实战)
    -   [pycharm 专业版](#pycharm-专业版-)
        -   [账号 地址 端口号 密码](#账号-地址-端口号-密码-)
        -   [创建连接](#创建连接)
        -   [选择服 AI 工作站运行环境](#选择服ai工作站运行环境-)
        -   [文件夹同步地址](#文件夹同步地址)
        -   [运行代码](#运行代码)
    -   [pycharm 社区版](#pycharm-社区版)
        _ [离线安装方式](#离线安装方式)
        _ [在线安装](#在线安装-)
        _ [配置连接](#配置连接-)
        _ [创建 ssh 连接](#创建ssh-连接)
        _ [输入地址用户名端口号](#输入地址用户名端口号-)
        _ [测试连接并同步文件](#测试连接并同步文件)
        _ [同步文件](#同步文件)
        _ [多文件同步](#多文件同步)
        _ [运行帮助](#运行帮助)
        _ [打开远程命令行 点击左下角命令行 选择更多 找到我们刚刚连接的命令行](#打开远程命令行--点击左下角命令行-选择更多-找到我们刚刚连接的命令行-)
        _ [如果需要运行 jupyter 项目 需要手动添加 `root` 环境运行方式](#如果需要运行jupyter-项目-需要手动添加-root-环境运行方式-)
        _ [删除 远程连接项目](#删除-远程连接项目-)
        <!-- TOC -->


 完整项目代码实战操作，请进入“人工智能教学实训平台”[https://www.lswai.com](https://www.lswai.com) 体验完整操作流程。

[![rtx4090.jpeg](https://i-blog.csdnimg.cn/img_convert/46bb006ecf29172f3af80c2ee260bc74.jpeg)](https://www.lswai.com)


## pycharm 专业版

#### 账号 地址 端口号 密码

打开网站  [https://www.lswai.com](https://www.lswai.com)，点击创建工作室之后 依据图片上 对应的地址端口号用户名进行连接

![img_0.png](https://i-blog.csdnimg.cn/img_convert/b608e58093ef23d1009dd3211e836ff5.png)

-   SSH 连接端口: `10020`
-   SSH 访问地址: `connect.cqa1.seetacloud.com`
-   SSH 用户名称: `root`
-   SSH 用户密码: `vTjOTHEY+R8g`

#### 创建连接

首先在 pycharm 左上角找到设置，找到 python 环境模块。

![img.png](https://i-blog.csdnimg.cn/img_convert/a9bf9967daab5daa62807f9f12631659.png)

选择 SSH 点击打开

![img_1.png](https://i-blog.csdnimg.cn/img_convert/17c84647216d292d1c3fc2082ee915a8.png)

从平台页面 复制对应的主机地址 用户名 端口号 和 密码 。

**网站密码位置图片**

选择新建， 然后将对应信息填入，点击下一步。

![img_2.png](https://i-blog.csdnimg.cn/img_convert/221e491d08dc5846a575830f29d1841b.png)

填入密码

![img_3.png](https://i-blog.csdnimg.cn/img_convert/214c329e7c5db83a8647a1c979f50d5f.png)

连接确认

![img_4.png](https://i-blog.csdnimg.cn/img_convert/341590cfeba9fc51d3a419bc472dd0fe.png)

#### 选择服 AI 工作站运行环境

**重点**

选择运行环境 ，点击现有，再点击 `...` ,打开远程 GPU 工作站环境列表。

![img_5.png](https://i-blog.csdnimg.cn/img_convert/68ce3af2ec67586515a9be425757804b.png)

找到 地址 `/root/miniconda3/bin/python3` , 点击确认

![img_6.png](https://i-blog.csdnimg.cn/img_convert/754c8bc1f53afdad2410cb61fb9555c8.png)

注意 不要勾选 以 `sudo` 方式运行 用户名账号本身就是 root 权限 追加 `sudo` 可能造成不必要的问题

选择自定义同步文件夹 ,修改默认同步位置到自定义位置, **切勿在 `/root` 以外的任何地方修改保存任何文件** 否则可能造成系统错误,产生任何不要的数据,计费时间损失.

在 `/root` 下属于系统盘可用空间 `30GB`, `/root/autodl-tmp` 下是一块 `50GB` 的高速 `SSD` 硬盘,可以进行存放数据.

#### 文件夹同步地址

**修改我们的文件同步地址**

![img_7.png](https://i-blog.csdnimg.cn/img_convert/0a5db2778e4b90c2f7fde818631ef933.png)

**我们新建一个我们的项目文件夹 `test2`,或者自定义文件夹.**

![img_8.png](https://i-blog.csdnimg.cn/img_convert/c09442f360c8868754287e6eeccee71a.png)

**选择刚刚我们创建的文件夹**

![img_9.png](https://i-blog.csdnimg.cn/img_convert/7e1c20d9ae9bbe033637bc7133706b43.png)

**点击确认**

![img_10.png](https://i-blog.csdnimg.cn/img_convert/5d17f287da0b1a59f2b50dab3313e9a4.png)

**最后确认一遍我们的设置,注意一定要勾选 `自动上传项目文件到服务器`**

![img_11.png](https://i-blog.csdnimg.cn/img_convert/cb309a18b631f90f4e9a4c2ae8ad673e.png)

**最后开始我们的工程创建**

![img_12.png](https://i-blog.csdnimg.cn/img_convert/08154cad80e1ab1ded20975f63d28689.png)

完成后 点击上下箭头可以看见实时的同步信息 ,等待同步完成以及 python 索引加载完成就可以开始我们的编程之旅了.

![img_13.png](https://i-blog.csdnimg.cn/img_convert/7a1ea4e9e64045a1e69a62410dffd4f8.png)

#### 运行代码

**运行代码**

右键运行可以直接启动远程环境 如图

![img_36.png](https://i-blog.csdnimg.cn/img_convert/38d9b9394d6c53117b4e50bc679753de.png)

---

## pycharm 社区版

> -   主要社区版需要依赖第三方插件,因第三方插件不受到 人工智能教学实训平台监管因此可能造成泄密等因素,因此请选择安全的第三方插件进行文件同步。

我们本次使用第三方插件 Source Synchronizer 作为示例。
供应商： 伊安-奥雷尔·福尔
插件 ID： org.wavescale.sourcesync
[许可证][1]

**离线安装地址**

[下载][下载] 兼容最低版本 2023.1x

[下载][下载2] 兼容最版本 2021.1 — 2022.2.5

#### 离线安装方式

打开设置 找到齿轮,选择从磁盘安装,

找到刚刚下载好的压缩包
![img_21.png](https://i-blog.csdnimg.cn/img_convert/c59513e8b4af4b990ffd486e5da832c3.png)

重启 IDE 即可安装完成

![img_22.png](https://i-blog.csdnimg.cn/img_convert/29a575d7e983d7fb465955b618006cab.png)

#### 在线安装

再插件主页搜索 Source Synchronizer 如图安装即可

![img_24.png](https://i-blog.csdnimg.cn/img_convert/a7fe230fe98be1508edc6cbbddc1c826.png)

#### 配置连接

从人工智能教学实训平台复制地址用户名

![img_38.png](https://i-blog.csdnimg.cn/img_convert/52d313fd508f184bcd36e0e51b767061.png)

点击 IDE 右上角 `Add Sourcesync Configuration...` 开始创建连接

![img_25.png](https://i-blog.csdnimg.cn/img_convert/9cbe68f1eff3413a862fc03938ee6346.png)

##### 创建 ssh 连接

![img_26.png](https://i-blog.csdnimg.cn/img_convert/b9a70d9bf007948bef9ad32ed10e173f.png)

##### 输入地址用户名端口号

从平台页面 复制对应的主机地址 用户名 端口号 和 密码 输入同步文件夹地址 `/root/autodl-tmp`

![img_37.png](https://i-blog.csdnimg.cn/img_convert/f75138e840ee2874ec77a8641cfa2565.png)

当如图所示 出现对应的情况则表示创建成功

![img_28.png](https://i-blog.csdnimg.cn/img_convert/def40dcb2d59cfdc7819dc4ceb540a4a.png)

##### 测试连接并同步文件

进入远程服务器命令行 粘贴 ssh 连接命令 输入 yes 和密码

![img_29.png](https://i-blog.csdnimg.cn/img_convert/f0bd8936b77f4ab7194bfcf594fde0e8.png)

![img_30.png](https://i-blog.csdnimg.cn/img_convert/d37648fe239d82c3e5b8545f107b850d.png)

###### 同步文件

首次同步文件请选择

![img_31.png](https://i-blog.csdnimg.cn/img_convert/bef2e0df7acfc07ec344115853aa049e.png)

同步完成后 即可运行文件

![img_32.png](https://i-blog.csdnimg.cn/img_convert/37b5932f101b4482cd4df6124fdc5713.png)

> -   注意,该插件需要手动同步文件,如果需要实时自动同步请选择专业版 IDE

###### 多文件同步

多选目标文件

![img_33.png](https://i-blog.csdnimg.cn/img_convert/280bb82569473167e11e0ec058aeae61.png)

同步完成后会发现文件夹内并未同步,该功能需要付费使用,该插件属于第三方服务插件谨慎购买

![img_34.png](https://i-blog.csdnimg.cn/img_convert/1708790574f28645706325936b873ae1.png)

如需文件夹及其子文件夹同步则请使用命令 scp 命令

```shell
scp -P 端口号 -r /本地文件夹路径 用户名@远程服务器IP:/远程目标路径

scp -P 49598 -r D:\test_3\test_3\test3s root@connect.cqa1.seetacloud.com:/root/autodl-tmp/test_3/test3s

```

![img_35.png](https://i-blog.csdnimg.cn/img_convert/609b3b184c21256a1d97b8c40b8bda2f.png)

---

### 运行帮助

#### 打开远程命令行 点击左下角命令行 选择更多 找到我们刚刚连接的命令行

![img_14.png](https://i-blog.csdnimg.cn/img_convert/8f4bbd55e58a118695f50cd05c156bd4.png)

![img_15.png](https://i-blog.csdnimg.cn/img_convert/8940abe9ec08b1846db52cd7ab36b55e.png)

#### 如果需要运行 jupyter 项目 需要手动添加 `root` 环境运行方式

![img_17.png](https://i-blog.csdnimg.cn/img_convert/e1590b59aa92458d0a1ce2a8c8c3ac50.png)

将对话框拉到底复制 `--allow-root` .

![img_18.png](https://i-blog.csdnimg.cn/img_convert/c57ab11a948683125babefb51e1772e5.png)

打开设置 输入 jupyter 选择 jupyter 服务器 追加启动命令,即可运行.

![img_19.png](https://i-blog.csdnimg.cn/img_convert/23ced2da2269a5667f684d917cc1efcc.png)

jupyter 内运行命令行 需要使用 `!/root/miniconda3/bin/python3 -m pip install` 才能调用 python 的 `pip`

如果出现提示没有包则可能原因为本地没有安装,如果远程安装有对应的包可以忽略

![img_16.png](https://i-blog.csdnimg.cn/img_convert/a7ccd3d20c3f58d5ba9f46b23dccb1cf.png)

#### 删除 远程连接项目

输入服务器 选择部署 删除我们刚刚创建的服务器即可

![img_20.png](https://i-blog.csdnimg.cn/img_convert/c76fb0d67c3457b38ecdb7066bf8cb7d.png)

[1]: https://plugins.jetbrains.com/files/7374/license.txt '许可证'
[下载]: https://plugins.jetbrains.com/plugin/download?rel=true&updateId=366303 '下载'
[下载2]: https://plugins.jetbrains.com/plugin/download?rel=true&updateId=209538 '下载'


 完整项目代码实战操作，请进入“人工智能教学实训平台”[https://www.lswai.com](https://www.lswai.com) 体验完整操作流程。

[![rtx4090.jpeg](https://i-blog.csdnimg.cn/img_convert/f47502608b60e1bb6024c6b4fda77022.jpeg)](https://www.lswai.com)
