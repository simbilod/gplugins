* `circuit.Package` `gplugins.parcirc.example.correct_pads`
* Generated by `vlsirtools.SpiceNetlister`
* 

.SUBCKT straight_e0091ba5
+ 1 
* No parameters

.ENDS

.SUBCKT straight_af1bc243
+ 1 
* No parameters

.ENDS

.SUBCKT straight_ea93caa4
+ 1 
* No parameters

.ENDS

.SUBCKT bend_euler_cross_sectionmetal3
+ 1 
* No parameters

.ENDS

.SUBCKT straight_55343d8d
+ 1 
* No parameters

.ENDS

.SUBCKT pad
+ 1 
* No parameters

.ENDS

.SUBCKT pads_correct
* No ports
* No parameters

x1
+ tl,tr 
+ pad
* No parameters


x2
+ tl,tr 
+ straight_ea93caa4
* No parameters


x3
+ tl,tr 
+ bend_euler_cross_sectionmetal3
* No parameters


x4
+ tl,tr 
+ straight_af1bc243
* No parameters


x5
+ tl,tr 
+ bend_euler_cross_sectionmetal3
* No parameters


x6
+ tl,tr 
+ pad
* No parameters


x7
+ tl,tr 
+ straight_e0091ba5
* No parameters


x8
+ bl,br 
+ pad
* No parameters


x9
+ bl,br 
+ straight_55343d8d
* No parameters


x10
+ bl,br 
+ pad
* No parameters


.ENDS

