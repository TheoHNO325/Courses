先记住一句话：**Git 里真正保存代码历史的是“提交（commit）”；分支、HEAD 都只是指向某个提交的“指针”；checkout 是切换指针和工作区；merge 是把两条提交线合到一起；PR 是 GitHub/GitLab 这类平台上的“合并申请单”。**

下面逐个解释。

---

## 1. 提交：Git 的核心

每次你执行 `git commit`，Git 都会生成一个提交节点，里面包含：

- 当前所有文件的快照
- 作者、时间、说明
- 父提交是谁

多个提交连起来，就像一条时间线：

```text
A --- B --- C --- D
```

`A` 是最早的提交，`D` 是最新提交。

---

## 2. 分支：指向提交的可移动指针

分支不是复制一份文件夹，它只是一个“指针”。

比如 `main` 指向 `D`：

```text
A --- B --- C --- D
                  ↑
                main
```

当你在 `main` 上再提交一个 `E`，`main` 会自动向前移动到 `E`：

```text
A --- B --- C --- D --- E
                        ↑
                      main
```

创建分支也很轻量：

```bash
git branch feature
```

这相当于又造了一个指针 `feature`，也指向当前提交：

```text
A --- B --- C --- D
                  ↑
                main
                feature
```

如果切换到 `feature` 并提交新内容，只有 `feature` 会移动，`main` 不动：

```text
A --- B --- C --- D --- E
                  ↑     ↑
                main  feature
```

所以，**分支的本质是：一条独立的开发线，背后只是一个可以移动的指针。**

---

## 3. checkout：切换你当前所在的位置

`git checkout` 的作用是：**把 HEAD 指向某个分支或提交，并把工作区文件更新成那个位置的样子。**

常见用法：

```bash
git checkout main          # 切换到 main 分支
git checkout feature       # 切换到 feature 分支
git checkout -b new-branch # 创建并切换到新分支
git checkout abc1234       # 切换到某个具体提交
```

其中：

- `git checkout main`：HEAD 指向 `main`，你就在 main 分支上工作。
- `git checkout abc1234`：HEAD 直接指向某个提交，不再指向任何分支。这就是你之前问的 **分离头指针（Detached HEAD）**。

因为 `checkout` 既管切分支，又管恢复文件，容易混。现在 Git 推荐拆成两个命令：

```bash
git switch main        # 切换分支
git restore file.txt   # 恢复文件
```

但很多教程和旧项目仍然用 `checkout`。

---

## 4. merge：把另一条分支合并到当前分支

`merge` 的意思是：**把另一个分支的提交历史，合并进你当前所在的分支。**

假设现在是这样：

```text
A --- B --- C    main
       \
        D --- E  feature
```

你站在 `main` 上，执行：

```bash
git merge feature
```

Git 会尝试把 `feature` 的改动合并到 `main`。

常见有两种结果：

### 快进合并 Fast-forward

如果 `main` 没有新提交，Git 直接把 `main` 指针移到 `feature` 的位置：

```text
A --- B --- C --- D --- E
                        ↑
                      main
                      feature
```

### 生成合并提交 Merge commit

如果两边都有新提交：

```text
A --- B --- C --- F    main
       \
        D --- E      feature
```

合并后会生成一个新的合并提交 `M`，它有两个父提交：

```text
A --- B --- C --- F --- M  main
       \               /
        D --- E ------
```

如果两个人改了同一文件的同一区域，就会产生 **冲突（conflict）**，需要手动解决，然后：

```bash
git add .
git commit
```

---

## 5. PR：Pull Request，平台上的合并请求

PR 不是 Git 本身的命令，而是 GitHub、GitLab、Gitee 等代码托管平台的功能。

它的意思是：**“我把我这个分支的改动推上去了，请求你把它合并到目标分支。”**

比如典型流程：

1. 从 `main` 创建 `feature` 分支：

```bash
git switch -c feature
```

2. 在 `feature` 上开发、提交：

```bash
git add .
git commit -m "add login"
```

3. 推送到远程：

```bash
git push -u origin feature
```

4. 在 GitHub/GitLab 网页上点 “New Pull Request”，选择：

```text
源分支：feature
目标分支：main
```

5. 写说明，请求别人 review。

6. 审查通过后，点击 “Merge”，平台会在后台执行合并，把 `feature` 合进 `main`。

所以：

- **PR 是“请求合并的工单”**
- **merge 是真正执行合并的动作**
- GitHub 叫 Pull Request，GitLab 叫 Merge Request，本质一样。

PR 的好处是：合并前可以代码审查、讨论、跑自动化测试，而不是直接往 `main` 推。

---

## 6. 把它们串起来

一个常见的团队工作流：

```text
main：稳定分支
feature：开发分支

1. git switch main
2. git pull
3. git switch -c feature
4. 写代码，git add，git commit
5. git push -u origin feature
6. 在平台开 PR：feature -> main
7. 同事 review，改问题
8. 点击 Merge
9. 本地切回 main，git pull，同步最新代码
```

对应关系：

| 概念 | 本质 | 作用 |
|---|---|---|
| 提交 commit | 代码快照节点 | 保存历史 |
| 分支 branch | 指向提交的指针 | 隔离开发线 |
| HEAD | 你当前在哪 | 决定新提交加到哪 |
| checkout | 切换 HEAD 和工作区 | 去某个分支或提交 |
| merge | 合并另一分支到当前分支 | 汇总代码历史 |
| PR | 平台上的合并请求 | 审查、讨论后再合并 |

---

## 最后用比喻总结

- **提交**：游戏存档点。
- **分支**：存档点上的书签，可以移动。
- **HEAD**：你当前正在读哪个书签。
- **checkout**：翻到某个书签。
- **merge**：把两条故事线合并成一条。
- **PR**：向管理员申请：“我想把我的草稿并入正式版，请先审核。”

理解 Git 的关键就是：**大多数操作不是在复制文件，而是在移动指针；真正的数据是提交。**