# GradCAM-of-PMG
This is an implementation of PMG, and I visualize the features abstracted by convs. It's helpful for me to understand the structure of PMG.  
Fork from https://github.com/PRIS-CV/PMG-Progressive-Multi-Granularity-Training , and I just trained on my own PC using CUB_200_2011. If you need the .pth, pls leave your email.  
The troubles I met have been noted in the code, some of them are Chinese. Hope it will be useful for you to understand.  
Meanwhile, I just practice to visualize the convs(https://github.com/IDayday/CAM-in-VGG-by-pytorch), so I do it.  

## Train Logs
```python
Step: 0 | Loss1: 0.171 | Loss2: 0.15630 | Loss3: 0.09787 | Loss_concat: 0.05488 | Loss: 0.480 | Acc: 100.000% (16/16)
Step: 50 | Loss1: 0.163 | Loss2: 0.08874 | Loss3: 0.06148 | Loss_concat: 0.03951 | Loss: 0.353 | Acc: 100.000% (816/816)
Step: 100 | Loss1: 0.170 | Loss2: 0.09134 | Loss3: 0.06481 | Loss_concat: 0.03879 | Loss: 0.365 | Acc: 100.000% (1616/1616)
Step: 150 | Loss1: 0.175 | Loss2: 0.08980 | Loss3: 0.06600 | Loss_concat: 0.03890 | Loss: 0.370 | Acc: 100.000% (2416/2416)
Step: 200 | Loss1: 0.174 | Loss2: 0.08840 | Loss3: 0.06554 | Loss_concat: 0.03959 | Loss: 0.367 | Acc: 100.000% (3216/3216)
Step: 250 | Loss1: 0.173 | Loss2: 0.08910 | Loss3: 0.06657 | Loss_concat: 0.04048 | Loss: 0.369 | Acc: 100.000% (4016/4016)
Step: 300 | Loss1: 0.173 | Loss2: 0.08668 | Loss3: 0.06656 | Loss_concat: 0.04027 | Loss: 0.366 | Acc: 100.000% (4816/4816)
Step: 350 | Loss1: 0.174 | Loss2: 0.08759 | Loss3: 0.06650 | Loss_concat: 0.04038 | Loss: 0.368 | Acc: 100.000% (5616/5616)
Step: 0 | Loss: 0.018 | Acc: 100.000% (3/3) |Combined Acc: 100.000% (3/3)
Step: 50 | Loss: 0.389 | Acc: 91.503% (140/153) |Combined Acc: 92.810% (142/153)
Step: 100 | Loss: 0.392 | Acc: 91.089% (276/303) |Combined Acc: 93.069% (282/303)
Step: 150 | Loss: 0.440 | Acc: 90.066% (408/453) |Combined Acc: 91.832% (416/453)
Step: 200 | Loss: 0.440 | Acc: 89.718% (541/603) |Combined Acc: 91.708% (553/603)
Step: 250 | Loss: 0.449 | Acc: 89.641% (675/753) |Combined Acc: 91.235% (687/753)
Step: 300 | Loss: 0.477 | Acc: 89.258% (806/903) |Combined Acc: 90.587% (818/903)
Step: 350 | Loss: 0.497 | Acc: 88.699% (934/1053) |Combined Acc: 89.839% (946/1053)
Step: 400 | Loss: 0.475 | Acc: 89.859% (1081/1203) |Combined Acc: 90.690% (1091/1203)
Step: 450 | Loss: 0.473 | Acc: 89.579% (1212/1353) |Combined Acc: 90.540% (1225/1353)
Step: 500 | Loss: 0.456 | Acc: 89.887% (1351/1503) |Combined Acc: 90.752% (1364/1503)
Step: 550 | Loss: 0.464 | Acc: 89.474% (1479/1653) |Combined Acc: 90.321% (1493/1653)
Step: 600 | Loss: 0.474 | Acc: 89.462% (1613/1803) |Combined Acc: 90.238% (1627/1803)
Step: 650 | Loss: 0.481 | Acc: 89.401% (1746/1953) |Combined Acc: 90.220% (1762/1953)
Step: 700 | Loss: 0.485 | Acc: 89.206% (1876/2103) |Combined Acc: 90.157% (1896/2103)
Step: 750 | Loss: 0.492 | Acc: 88.859% (2002/2253) |Combined Acc: 89.791% (2023/2253)
Step: 800 | Loss: 0.496 | Acc: 88.847% (2135/2403) |Combined Acc: 89.846% (2159/2403)
Step: 850 | Loss: 0.489 | Acc: 88.954% (2271/2553) |Combined Acc: 89.777% (2292/2553)
Step: 900 | Loss: 0.491 | Acc: 88.938% (2404/2703) |Combined Acc: 89.715% (2425/2703)
Step: 950 | Loss: 0.487 | Acc: 88.889% (2536/2853) |Combined Acc: 89.590% (2556/2853)
Step: 1000 | Loss: 0.486 | Acc: 88.811% (2667/3003) |Combined Acc: 89.444% (2686/3003)
Step: 1050 | Loss: 0.494 | Acc: 88.804% (2800/3153) |Combined Acc: 89.407% (2819/3153)
Step: 1100 | Loss: 0.500 | Acc: 88.495% (2923/3303) |Combined Acc: 89.161% (2945/3303)
Step: 1150 | Loss: 0.501 | Acc: 88.561% (3058/3453) |Combined Acc: 89.198% (3080/3453)
Step: 1200 | Loss: 0.503 | Acc: 88.565% (3191/3603) |Combined Acc: 89.176% (3213/3603)
Step: 1250 | Loss: 0.501 | Acc: 88.569% (3324/3753) |Combined Acc: 89.209% (3348/3753)
Step: 1300 | Loss: 0.499 | Acc: 88.573% (3457/3903) |Combined Acc: 89.213% (3482/3903)
Step: 1350 | Loss: 0.503 | Acc: 88.527% (3588/4053) |Combined Acc: 89.193% (3615/4053)
Step: 1400 | Loss: 0.506 | Acc: 88.342% (3713/4203) |Combined Acc: 89.032% (3742/4203)
Step: 1450 | Loss: 0.505 | Acc: 88.169% (3838/4353) |Combined Acc: 88.881% (3869/4353)
Step: 1500 | Loss: 0.511 | Acc: 88.052% (3965/4503) |Combined Acc: 88.807% (3999/4503)
Step: 1550 | Loss: 0.506 | Acc: 88.180% (4103/4653) |Combined Acc: 88.953% (4139/4653)
Step: 1600 | Loss: 0.508 | Acc: 88.174% (4235/4803) |Combined Acc: 88.924% (4271/4803)
Step: 1650 | Loss: 0.511 | Acc: 88.128% (4365/4953) |Combined Acc: 88.855% (4401/4953)
Step: 1700 | Loss: 0.510 | Acc: 88.125% (4497/5103) |Combined Acc: 88.811% (4532/5103)
Step: 1750 | Loss: 0.508 | Acc: 88.121% (4629/5253) |Combined Acc: 88.825% (4666/5253)
Step: 1800 | Loss: 0.503 | Acc: 88.247% (4768/5403) |Combined Acc: 89.025% (4810/5403)
Step: 1850 | Loss: 0.501 | Acc: 88.259% (4901/5553) |Combined Acc: 89.069% (4946/5553)
Step: 1900 | Loss: 0.500 | Acc: 88.287% (5035/5703) |Combined Acc: 89.076% (5080/5703)
```
* The up lines are acc in trainsets, others are acc in testsets. It has been closed to the SOTA accuracy.(I trained 80 epochs)

## GradCAM Results  

### Original image  
1.jpg|9.jpg
:---:|:---:
<img src="https://github.com/IDayday/GradCAM-of-PMG/blob/main/original%20img/1.jpg" width="150" alt="1.jpg">|<img src="https://github.com/IDayday/GradCAM-of-PMG/blob/main/original%20img/9.jpg" width="150" alt="9.jpg">

### Convs visualization

conv_block1|conv_block2|conv_block3|conv_block_concat
:---:|:---:|:---:|:---:
<img src="https://github.com/IDayday/GradCAM-of-PMG/blob/main/cam%20results/conv_block1_class003_1.jpg" width="150" alt="1.jpg">|<img src="https://github.com/IDayday/GradCAM-of-PMG/blob/main/cam%20results/conv_block2_class003_1.jpg" width="150" alt="1.jpg">|<img src="https://github.com/IDayday/GradCAM-of-PMG/blob/main/conv_block3_class003_1.jpg" width="150" alt="1.jpg">|<img src="https://github.com/IDayday/GradCAM-of-PMG/blob/main/cam%20results/conv_block_concat_class003_1.jpg" width="150" alt="1.jpg">
<img src="https://github.com/IDayday/GradCAM-of-PMG/blob/main/cam%20results/conv_block1_class003_9.jpg" width="150" alt="9.jpg">|<img src="https://github.com/IDayday/GradCAM-of-PMG/blob/main/cam%20results/conv_block2_class003_9.jpg" width="150" alt="9.jpg">|<img src="https://github.com/IDayday/GradCAM-of-PMG/blob/main/cam%20results/conv_block3_class003_9.jpg" width="150" alt="9.jpg">|<img src="https://github.com/IDayday/GradCAM-of-PMG/blob/main/cam%20results/conv_block_concat_class003_9.jpg" width="150" alt="9.jpg">
