---
layout: post
title: 샘플 포스트
description: "Just about everything you'll need to style in the theme: headings, paragraphs, blockquotes, tables, code blocks, and more."
modified: 2014-12-24
tags: [sample post]
category: samples
image:
  feature: abstract-3.jpg
  credit:
  creditlink:
---

본 포스트에서는 기본적인 MarkDown Language를 사용한 태마 사용법에 대해서 알아보겠습니다.

다음 코드와 같이 내용 앞에 '#'을 붙이면 해당 내용을 제목(Head Line)으로 표시할 수 있습니다.
각 #의 개수는 Html 코드의 h1, h2, h3, h4, h5 에 대응됩니다.

``` apiblueprint
# Heading 1

## Heading 2

### Heading 3

#### Heading 4

##### Heading 5

###### Heading 6
```

# Heading 1

## Heading 2

### Heading 3

#### Heading 4

##### Heading 5

###### Heading 6

### Body text

다음 코드를 사용하면, 아래와 같이 그림을 추가할 수 있습니다. 아래 링크를 활용하면 깃허브 페이지에 이미지를 쉽게 추가할 수 있다고 합니다.
[Github 마크다운 작성 시 이미지 업로드 꿀팁](https://ahribori.com/article/5a03bcfd6c9eef13d882e29a)

``` apiblueprint
![3953273590_704e3899d5_m](https://user-images.githubusercontent.com/14049664/37239363-9b099be8-247d-11e8-8bb2-13376f896844.jpg)
{: .image-right}
```

![3953273590_704e3899d5_m](https://user-images.githubusercontent.com/14049664/37239363-9b099be8-247d-11e8-8bb2-13376f896844.jpg)
{: .image-right}

*This is emphasized*. Donec faucibus. Nunc iaculis suscipit dui. 53 = 125. Water is H<sub>2</sub>O. Nam sit amet sem. Aliquam libero nisi, imperdiet at, tincidunt nec, gravida vehicula, nisl. The New York Times <cite>(That’s a citation)</cite>. <u>Underline</u>. Maecenas ornare tortor. Donec sed tellus eget sapien fringilla nonummy. Mauris a ante. Suspendisse quam sem, consequat at, commodo vitae, feugiat in, nunc. Morbi imperdiet augue quis tellus.

HTML and <abbr title="cascading stylesheets">CSS<abbr> are our tools. Mauris a ante. Suspendisse quam sem, consequat at, commodo vitae, feugiat in, nunc. Morbi imperdiet augue quis tellus. Praesent mattis, massa quis luctus fermentum, turpis mi volutpat justo, eu volutpat enim diam eget metus.

### Blockquotes

``` apiblueprint
> Lorem ipsum dolor sit amet, test link adipiscing elit. Nullam dignissim convallis est. Quisque aliquam.
```

> Lorem ipsum dolor sit amet, test link adipiscing elit. Nullam dignissim convallis est. Quisque aliquam.

## List Types

### Ordered Lists

1. Item one
   1. sub item one
   2. sub item two
   3. sub item three
2. Item two

### Unordered Lists

* Item one
* Item two
* Item three

## Tables

| Header1 | Header2 | Header3 |
|:--------|:-------:|--------:|
| cell1   | cell2   | cell3   |
| cell4   | cell5   | cell6   |
|----
| cell1   | cell2   | cell3   |
| cell4   | cell5   | cell6   |
|=====
| Foot1   | Foot2   | Foot3
{: rules="groups"}

## Code Snippets

Syntax highlighting via Rouge

```css
#container {
  float: left;
  margin: 0 -240px 0 0;
  width: 100%;
}
```

Non Pygments code example

    <div id="awesome">
        <p>This is great isn't it?</p>
    </div>

## Buttons

Make any link standout more when applying the `.btn` class.

```html
<a href="#" class="btn btn-success">Success Button</a>
```

<div markdown="0"><a href="#" class="btn">Primary Button</a></div>
<div markdown="0"><a href="#" class="btn btn-success">Success Button</a></div>
<div markdown="0"><a href="#" class="btn btn-warning">Warning Button</a></div>
<div markdown="0"><a href="#" class="btn btn-danger">Danger Button</a></div>
<div markdown="0"><a href="#" class="btn btn-info">Info Button</a></div>
