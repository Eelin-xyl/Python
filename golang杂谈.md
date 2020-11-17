# golang杂谈

## 安装

苹果用户选择pkg安装，默认路径为`/usr/local/go`

## 第一个golang

Just hello world, you see it.

vim hello.go

```go
package main

import "fmt"

func main(){
    fmt.Println("hello world")
}
```

执行go run hello.go

## 变量类型

### 整型

区分有符号整型和无符号整型

uint8等价于byte

### 浮点数

float64/32

### 变量类型推导

i:=10

### 整型字符串转换 

```go
func changeType() {
	num:=10
	name:="xiaoyuyu"
	//	整型转字符串
	itob := strconv.Itoa(num)
	//	字符串转整型
	atob,_:= strconv.Atoi(name)
	fmt.Println(reflect.TypeOf(itob))
	fmt.Println(reflect.TypeOf(atob))
}
```

## 地址，指针

& and *

## 常量

```go
func constTest() {
	const name = "xiaoyuyu"
	const (
		one = iota + 1
		two
		three
	)
	fmt.Println(one)
	fmt.Println(two)
	fmt.Println(three)
}
```

## 字符串的工具包Strings

## switch

注意fallthrough的用法

## 切片Slice

切片的三种声明方式

```go
package main

import "fmt"

// slice 的三种声明方式
func sliceTest()  {
	// 方式1
	slice1 := []string{"1","3","1","4","o"}
	fmt.Println(slice1)	// [1 3 1 4 o]

	// 方式2
	slice2 := make([]string,4,8)	// type,len,cap
	slice2 = append(slice2, "a")
	slice2 = append(slice2,"b","c")	// [    a b c]
	fmt.Println(slice2)

	// 方式3，从数组导出
	array := [5]string{"a","b","c","d","e"}
	slice3 := array[2:5]	// 切片包括start,不包括end
	fmt.Println(slice3)	// [c d e]
	slice3[1] = "x"
	fmt.Println(array)	// [a b c x e], 对应的array也发生修改，证明slice底层就是array
}

func main() {
	sliceTest()
}

```

小技巧：在创建新切片的时候，最好要让新切片的长度和容量一样，这样在追加操作的时候就会生成新的底层数组，从而和原有数组分离，就不会因为共用底层数组导致修改内容的时候影响多个切片

## map

```go
// 测试map
func mapTest()  {
	nameAge := make(map[string]int)
	nameAge["louwin"] = 23
	nameAge["xiaoyuyu"] = 22
	//测试 for range
	fmt.Println("map len is",len(nameAge))
	for k,v:=range nameAge{
		fmt.Println("Key is",k,",Value is",v)
	}
	delete(nameAge,"飞雪无情")	// 删除键值对，不存在也可以调用
	age,ok:=nameAge["king"]	// age 是返回的 Value，ok 用来标记该 Key 是否存在
	if ok {
		fmt.Println(age)
	}
}

func main() {
	sliceTest()
	mapTest()
}
```

## String 和 []byte

字符串 string 也是一个不可变的字节序列，所以可以直接转为字节切片 []byte

```go
s:="Hello飞雪无情"
bs:=[]byte(s)
fmt.Println(bs)
fmt.Println(s[0],s[1],s[15])
```

可以看到如果是中文的话，调用这个方式要小心，因为byte和unicode有区别，使用range即可隐式的调用中文，如下

```go
for i,r:=range s{
    fmt.Println(i,r)
}
```

## 函数

### 基本函数结构

注：函数可以多值返回

```go
func sum(a, b int) (int,error){
    if a<0 || b<0 {
        return 0,errors.New("a或者b不能是负数")
    }
    return a + b,nil
}
```

### 可变参数

只要在参数类型前加三个点 … 即可

```go
func sum1(params ...int) int {
    sum := 0
    for _, i := range params {
        sum += i
    }
    return sum
}
```

### 包级函数

这里可以先记住：

1. 函数名称首字母小写代表私有函数，只有在同一个包中才可以被调用。
2. 函数名称首字母大写代表公有函数，不同的包也可以调用。
3. 任何一个函数都会从属于一个包。

### 匿名函数

```go
func main() {
    sum2 := func(a, b int) int {
        return a + b
    }
    fmt.Println(sum2(1, 2))
}
```

### 闭包

```go
func main() {
    cl:=colsure()
    fmt.Println(cl())
    fmt.Println(cl())
    fmt.Println(cl())
}
func colsure() func() int {
    i:=0
    return func() int {
        i++
        return i
    }
}
```

在 Go 语言中，函数也是一种类型，它也可以被用来声明函数类型的变量、参数或者作为另一个函数的返回值类型

## 方法

多了一个函数的接收者

一般用在类中，如果需要改变对应类中的元素，需要接收者是指针

```go
func (age *Age) Modify(){
    *age = Age(30)
}
```

## 结构体struct

结构体的基本使用

```go
package main

import "fmt"

type address struct {
	province string
	city string
}

type person struct {
	name string
	age uint
	addr address
}



func main() {
	var p person
	fmt.Println(p)	//	{ 0 { }}
	p2:=person{name:"xiaoyuyu",age:18,addr:address{
		province: "shanghai",
		city:     "shanghai",
	}}	//	{xiaoyuyu 18 {shanghai shanghai}}
	fmt.Println(p2)
}

```

### 工厂函数

工厂函数一般用于创建自定义的结构体，便于使用者调用

```go
func NewPerson(name string) *person {
    return &person{name:name}
}

p1:=NewPerson("张三")
```

## 接口interface

一个对象只要全部实现了接口中的方法，那么就实现了这个接口。换句话说，接口就是一个**需要实现的方法列表**

```go
package main

import "fmt"

// Sayer 接口
type Sayer interface {
	say()
}

type dog struct {}
type cat struct {}

// dog实现了Sayer接口
func (d dog) say() {
	fmt.Println("汪汪汪")
}

// cat实现了Sayer接口
func (c cat) say() {
	fmt.Println("喵喵喵")
}

func say(s Sayer){
	s.say()
}

func main() {
	animal:=dog{}
	animal2:=cat{}
	say(animal)	//	汪汪汪
	say(animal2)	//	喵喵喵
}

```

因为`Sayer`接口里只有一个`say`方法，所以我们只需要给`dog`和`cat `分别实现`say`方法就可以实现`Sayer`接口了

一个例子感受到接口的魅力

| 方法接收者  |  实现接口类型   |
| :---------: | :-------------: |
| (p person)  | *person和person |
| (p *person) |     *person     |

## 组合(golang没有继承)

结构体和接口都可以组合，组合后分为内外部类型，外部类型可以使用内部类型的方法，也可以重写内部类型的方法

## 类型断言

golang里很神奇的一种操作

```go
func main() {
	var x interface{}
	x = "Hello 沙河"
	v, ok := x.(string)
	if ok {
		fmt.Println(v)
	} else {
		fmt.Println("类型断言失败")
	}
}
```

