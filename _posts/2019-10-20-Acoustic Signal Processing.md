---
layout: post
title: Acoustic Signal Processing (Source Enhancement, Localization, Detection) in Low SNR environments
description: "results of my research"
modified: 2019-10-20
tags: [Technical Guide]
category: Research
image:
  feature: paper.jpg
  credit:
  creditlink:
---

## Enhancement results

<div>
<strong stype="font-size: 1.17em;"> 1. SNR -17.09 (1m between sound source and hovering multi-rotor UAV) </strong>
</div>
<div style="width: 49%; display: inline-block;">
<ul>
<li> sound enhanced by RICEN </li>
</ul>
<audio controls loop style="width: 100%">
    <source src="/audios/dist_1/estimated/timit_0-0.wav" type="audio/wav">
</audio>
</div>
<div style="width: 50%; display: inline-block;">
<ul>
<li> noisy sound </li>
</ul>
<audio controls loop style="width: 100%">
    <source src="/audios/dist_1/noisy/timit_0-0.wav" type="audio/wav">
</audio>
</div>
<div style="width: 49%; display: inline-block;">
<ul>
<li> noise sound </li>
</ul>
<audio controls loop style="width: 100%">
    <source src="/audios/dist_1/noise/timit_0-0.wav" type="audio/wav">
</audio>
</div>
<div style="width: 50%; display: inline-block;">
<ul>
<li> clean sound </li>
</ul>
<audio controls loop style="width: 100%">
    <source src="/audios/dist_1/clean/timit_0-0.wav" type="audio/wav">
</audio>
</div>

<br />

<div>
<strong stype="font-size: 1.17em;"> 2. SNR -21.37 (5m between sound source and rotating multi-rotor UAV) </strong>
</div>
<div style="width: 49%; display: inline-block;">
<ul>
<li> sound enhanced by RICEN </li>
</ul>
<audio controls loop style="width: 100%">
    <source src="/audios/dist_5/estimated/timit_95-100.wav" type="audio/wav">
</audio>
</div>
<div style="width: 50%; display: inline-block;">
<ul>
<li> noisy sound </li>
</ul>
<audio controls loop style="width: 100%">
    <source src="/audios/dist_5/noisy/timit_95-100.wav" type="audio/wav">
</audio>
</div>
<div style="width: 49%; display: inline-block;">
<ul>
<li> noise sound </li>
</ul>
<audio controls loop style="width: 100%">
    <source src="/audios/dist_5/noise/timit_95-100.wav" type="audio/wav">
</audio>
</div>
<div style="width: 50%; display: inline-block;">
<ul>
<li> clean sound </li>
</ul>
<audio controls loop style="width: 100%">
    <source src="/audios/dist_5/clean/timit_95-100.wav" type="audio/wav">
</audio>
</div>

<br />

<div>
<strong stype="font-size: 1.17em;"> 3. SNR -27.14 (10m between sound source and shifting(moving) multi-rotor UAV) </strong>
</div>
<div style="width: 49%; display: inline-block;">
<ul>
<li> sound enhanced by RICEN </li>
</ul>
<audio controls loop style="width: 100%">
    <source src="/audios/dist_10/estimated/timit_344-80.wav" type="audio/wav">
</audio>
</div>
<div style="width: 50%; display: inline-block;">
<ul>
<li> noisy sound </li>
</ul>
<audio controls loop style="width: 100%">
    <source src="/audios/dist_10/noisy/timit_344-80.wav" type="audio/wav">
</audio>
</div>
<div style="width: 49%; display: inline-block;">
<ul>
<li> noise sound </li>
</ul>
<audio controls loop style="width: 100%">
    <source src="/audios/dist_10/noise/timit_344-80.wav" type="audio/wav">
</audio>
</div>
<div style="width: 50%; display: inline-block;">
<ul>
<li> clean sound </li>
</ul>
<audio controls loop style="width: 100%">
    <source src="/audios/dist_10/clean/timit_344-80.wav" type="audio/wav">
</audio>
</div>

## Localization $ Detection results

+ noisy Log Magnitude Spectra
+ clean Log Magnitude Spectra
+ estimated Direction of Arrival
+ detected voice activity

<strong stype="font-size: 1.17em;"> 1. SNR -21.37 (5m between sound source and rotating multi-rotor UAV) </strong>

<figure>
	<img src="/audios/dist_5/timit_95-100.png" alt="">
	<figcaption>direction of arrive at 100 degree</figcaption>
</figure>

<strong stype="font-size: 1.17em;"> 2. SNR -25.05 (5m between sound source and hovering multi-rotor UAV) </strong>

<figure>
	<img src="/audios/dist_5/timit_268-160.png" alt="">
	<figcaption>direction of arrive at 160 degree</figcaption>
</figure>

<strong stype="font-size: 1.17em;"> 3. SNR -27.14 (10m between sound source and shifting(moving) multi-rotor UAV) </strong>

<figure>
	<img src="/audios/dist_10/timit_344-80.png" alt="">
	<figcaption>direction of arrive at 80 degree</figcaption>
</figure>

<strong stype="font-size: 1.17em;"> 4. SNR -35.93 (15m between sound source and shifting(moving) multi-rotor UAV) </strong>

<figure>
	<img src="/audios/dist_15/timit_591-20.png" alt="">
	<figcaption>direction of arrive at 20 degree</figcaption>
</figure>