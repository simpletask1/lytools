## DOS 命令
    进入指定盘 : c:或d:
    dir : ls
    md : mkdir
    rd 删除目录
    cd
    cd..
    cd\ : cd ~
    del 删除文件
    exit 退出DOS命令行

## C++
### 优先学习
    * makefile编译流程；cmake（CmakeLists）
    * 工程结构：.h中声明，.cc中定义
    * 变量作用域；函数；指针；面向对象；opencv；
### 笔记及语法示例
#### 函数
    函数声明告诉编译器函数的`名称`、`返回类型`和`参数`
    函数定义提供了函数的实际主体
    
    当您在一个源文件中定义函数且在另一个文件中调用函数时，函数声明是必需的
    在这种情况下，您应该在调用函数的文件顶部声明函数
```c++
typedef type new_name;

enum type_name{a, b, c} x;
x = a;

class MyClass{
  public:
    int a = 1;
  private:
    // 成员函数声明
    bool init();
  protected:
    char c;
};

// 成员函数定义
bool MyClass::init(void){
  std::cout << 'start init' << std::endl
}

namespace my_namespace{
 // 变量或函数的声明
}
// 命名空间可以不连续，上述声明可以是定义一个新的命名空间，
// 也可以是为已有的命名空间增加新的元素
```



