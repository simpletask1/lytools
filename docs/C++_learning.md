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

## C++编译
    每个源文件都应该对应于一个中间目标文件（O文件或是OBJ文件）
    中间目标文件打包为库文件（windows下的.lib 文件，在UNIX下是Archive File，也就是.a 文件）
    中间目标文件链接目标程序，生成执行文件
## C++语法
### 优先学习
    * makefile编译流程；cmake（CmakeLists）
    * 工程结构：.cc中定义，.h中声明
    * 函数；指针；面向对象；
    * opencv2/opencv.hpp；nlohmann/json.hpp；boost/filesystem.hpp
### 笔记及示例
#### 函数
    函数声明告诉编译器函数的`名称`、`返回类型`和`参数`
    函数定义提供了函数的实际主体
    
    当您在一个源文件中定义函数且在另一个文件中调用函数时，函数声明是必需的
    在这种情况下，您应该在调用函数的文件顶部声明函数
#### 字符
    字符：单引号（L开头是宽字符，如L'x'）
    字符串：双引号
#### 常量
    两种定义方式：
    #define预处理器
    const关键字
#### 指针
    1. *的用法：
    指针声明：int* p; 或 int *p;
    复合指针：int** p; 或 int **p;
    解引用：x = *p;
    2. &的用法：
    求地址：p = &x; //把x的地址赋给指针p
    传引用：void func(int& r){} //传入的是变量的引用，直接对传入的变量进行操作
    3. p->a 就是 (*p).a
    
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

// 命名空间可以不连续，下述声明可以是定义一个新的命名空间，
// 也可以是为已有的命名空间增加新的元素
namespace my_namespace{
 // 变量或函数的声明
} // namespace my_namespace

```



