??
??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

s
FakeQuantWithMinMaxVars

inputs
min
max
outputs"
num_bitsint"
narrow_rangebool( 
}
!FakeQuantWithMinMaxVarsPerChannel

inputs
min
max
outputs"
num_bitsint"
narrow_rangebool( 
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
?
!quantize_layer/quantize_layer_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!quantize_layer/quantize_layer_min
?
5quantize_layer/quantize_layer_min/Read/ReadVariableOpReadVariableOp!quantize_layer/quantize_layer_min*
_output_shapes
: *
dtype0
?
!quantize_layer/quantize_layer_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *2
shared_name#!quantize_layer/quantize_layer_max
?
5quantize_layer/quantize_layer_max/Read/ReadVariableOpReadVariableOp!quantize_layer/quantize_layer_max*
_output_shapes
: *
dtype0
?
quantize_layer/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *.
shared_namequantize_layer/optimizer_step
?
1quantize_layer/optimizer_step/Read/ReadVariableOpReadVariableOpquantize_layer/optimizer_step*
_output_shapes
: *
dtype0
?
quant_conv2d/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *,
shared_namequant_conv2d/optimizer_step
?
/quant_conv2d/optimizer_step/Read/ReadVariableOpReadVariableOpquant_conv2d/optimizer_step*
_output_shapes
: *
dtype0
?
quant_conv2d/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namequant_conv2d/kernel_min

+quant_conv2d/kernel_min/Read/ReadVariableOpReadVariableOpquant_conv2d/kernel_min*
_output_shapes
:*
dtype0
?
quant_conv2d/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_namequant_conv2d/kernel_max

+quant_conv2d/kernel_max/Read/ReadVariableOpReadVariableOpquant_conv2d/kernel_max*
_output_shapes
:*
dtype0
?
 quant_conv2d/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" quant_conv2d/post_activation_min
?
4quant_conv2d/post_activation_min/Read/ReadVariableOpReadVariableOp quant_conv2d/post_activation_min*
_output_shapes
: *
dtype0
?
 quant_conv2d/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *1
shared_name" quant_conv2d/post_activation_max
?
4quant_conv2d/post_activation_max/Read/ReadVariableOpReadVariableOp quant_conv2d/post_activation_max*
_output_shapes
: *
dtype0
?
"quant_max_pooling2d/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *3
shared_name$"quant_max_pooling2d/optimizer_step
?
6quant_max_pooling2d/optimizer_step/Read/ReadVariableOpReadVariableOp"quant_max_pooling2d/optimizer_step*
_output_shapes
: *
dtype0
?
quant_dropout/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namequant_dropout/optimizer_step
?
0quant_dropout/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dropout/optimizer_step*
_output_shapes
: *
dtype0
?
quant_flatten/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *-
shared_namequant_flatten/optimizer_step
?
0quant_flatten/optimizer_step/Read/ReadVariableOpReadVariableOpquant_flatten/optimizer_step*
_output_shapes
: *
dtype0
?
quant_dense/optimizer_stepVarHandleOp*
_output_shapes
: *
dtype0*
shape: *+
shared_namequant_dense/optimizer_step
?
.quant_dense/optimizer_step/Read/ReadVariableOpReadVariableOpquant_dense/optimizer_step*
_output_shapes
: *
dtype0
?
quant_dense/kernel_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namequant_dense/kernel_min
y
*quant_dense/kernel_min/Read/ReadVariableOpReadVariableOpquant_dense/kernel_min*
_output_shapes
: *
dtype0
?
quant_dense/kernel_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *'
shared_namequant_dense/kernel_max
y
*quant_dense/kernel_max/Read/ReadVariableOpReadVariableOpquant_dense/kernel_max*
_output_shapes
: *
dtype0
?
quant_dense/post_activation_minVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!quant_dense/post_activation_min
?
3quant_dense/post_activation_min/Read/ReadVariableOpReadVariableOpquant_dense/post_activation_min*
_output_shapes
: *
dtype0
?
quant_dense/post_activation_maxVarHandleOp*
_output_shapes
: *
dtype0*
shape: *0
shared_name!quant_dense/post_activation_max
?
3quant_dense/post_activation_max/Read/ReadVariableOpReadVariableOpquant_dense/post_activation_max*
_output_shapes
: *
dtype0
`
beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_1
Y
beta_1/Read/ReadVariableOpReadVariableOpbeta_1*
_output_shapes
: *
dtype0
`
beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namebeta_2
Y
beta_2/Read/ReadVariableOpReadVariableOpbeta_2*
_output_shapes
: *
dtype0
^
decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namedecay
W
decay/Read/ReadVariableOpReadVariableOpdecay*
_output_shapes
: *
dtype0
n
learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namelearning_rate
g
!learning_rate/Read/ReadVariableOpReadVariableOplearning_rate*
_output_shapes
: *
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
n
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameconv2d/bias
g
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes
:*
dtype0
~
conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameconv2d/kernel
w
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*&
_output_shapes
:
*
dtype0
l

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
e
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes
:*
dtype0
u
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*
shared_namedense/kernel
n
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel*
_output_shapes
:	?*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
|
Adam/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/m
u
&Adam/conv2d/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/m*
_output_shapes
:*
dtype0
?
Adam/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/conv2d/kernel/m
?
(Adam/conv2d/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/m*&
_output_shapes
:
*
dtype0
z
Adam/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/m
s
%Adam/dense/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense/bias/m*
_output_shapes
:*
dtype0
?
Adam/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*$
shared_nameAdam/dense/kernel/m
|
'Adam/dense/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/m*
_output_shapes
:	?*
dtype0
|
Adam/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*#
shared_nameAdam/conv2d/bias/v
u
&Adam/conv2d/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/bias/v*
_output_shapes
:*
dtype0
?
Adam/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*%
shared_nameAdam/conv2d/kernel/v
?
(Adam/conv2d/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2d/kernel/v*&
_output_shapes
:
*
dtype0
z
Adam/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*"
shared_nameAdam/dense/bias/v
s
%Adam/dense/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense/bias/v*
_output_shapes
:*
dtype0
?
Adam/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*$
shared_nameAdam/dense/kernel/v
|
'Adam/dense/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense/kernel/v*
_output_shapes
:	?*
dtype0

NoOpNoOp
?I
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?I
value?IB?I B?I
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api

signatures
?
quantize_layer_min
quantize_layer_max
quantizer_vars
optimizer_step
	variables
trainable_variables
regularization_losses
	keras_api
?
	layer
optimizer_step
_weight_vars

kernel_min

kernel_max
_quantize_activations
post_activation_min
post_activation_max
_output_quantizers
	variables
trainable_variables
 regularization_losses
!	keras_api
?
	"layer
#optimizer_step
$_weight_vars
%_quantize_activations
&_output_quantizers
'	variables
(trainable_variables
)regularization_losses
*	keras_api
?
	+layer
,optimizer_step
-_weight_vars
._quantize_activations
/_output_quantizers
0	variables
1trainable_variables
2regularization_losses
3	keras_api
?
	4layer
5optimizer_step
6_weight_vars
7_quantize_activations
8_output_quantizers
9	variables
:trainable_variables
;regularization_losses
<	keras_api
?
	=layer
>optimizer_step
?_weight_vars
@
kernel_min
A
kernel_max
B_quantize_activations
Cpost_activation_min
Dpost_activation_max
E_output_quantizers
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
?

Jbeta_1

Kbeta_2
	Ldecay
Mlearning_rate
NiterOm?Pm?Qm?Rm?Ov?Pv?Qv?Rv?
?
0
1
2
O3
P4
5
6
7
8
9
#10
,11
512
Q13
R14
>15
@16
A17
C18
D19

O0
P1
Q2
R3
 
?
Slayer_metrics
Tmetrics
	variables

Ulayers
Vlayer_regularization_losses
	trainable_variables

regularization_losses
Wnon_trainable_variables
 
yw
VARIABLE_VALUE!quantize_layer/quantize_layer_minBlayer_with_weights-0/quantize_layer_min/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE!quantize_layer/quantize_layer_maxBlayer_with_weights-0/quantize_layer_max/.ATTRIBUTES/VARIABLE_VALUE

min_var
max_var
qo
VARIABLE_VALUEquantize_layer/optimizer_step>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

0
1
2
 
 
?
Xlayer_metrics
Ymetrics
	variables

Zlayers
[layer_regularization_losses
trainable_variables
regularization_losses
\non_trainable_variables
h

Pkernel
Obias
]	variables
^trainable_variables
_regularization_losses
`	keras_api
om
VARIABLE_VALUEquant_conv2d/optimizer_step>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

a0
ge
VARIABLE_VALUEquant_conv2d/kernel_min:layer_with_weights-1/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
ge
VARIABLE_VALUEquant_conv2d/kernel_max:layer_with_weights-1/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
yw
VARIABLE_VALUE quant_conv2d/post_activation_minClayer_with_weights-1/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE quant_conv2d/post_activation_maxClayer_with_weights-1/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
 
1
O0
P1
2
3
4
5
6

O0
P1
 
?
blayer_metrics
cmetrics
	variables

dlayers
elayer_regularization_losses
trainable_variables
 regularization_losses
fnon_trainable_variables
R
g	variables
htrainable_variables
iregularization_losses
j	keras_api
vt
VARIABLE_VALUE"quant_max_pooling2d/optimizer_step>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

#0
 
 
?
klayer_metrics
lmetrics
'	variables

mlayers
nlayer_regularization_losses
(trainable_variables
)regularization_losses
onon_trainable_variables
R
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
pn
VARIABLE_VALUEquant_dropout/optimizer_step>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

,0
 
 
?
tlayer_metrics
umetrics
0	variables

vlayers
wlayer_regularization_losses
1trainable_variables
2regularization_losses
xnon_trainable_variables
R
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
pn
VARIABLE_VALUEquant_flatten/optimizer_step>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE
 
 
 

50
 
 
?
}layer_metrics
~metrics
9	variables

layers
 ?layer_regularization_losses
:trainable_variables
;regularization_losses
?non_trainable_variables
l

Rkernel
Qbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
nl
VARIABLE_VALUEquant_dense/optimizer_step>layer_with_weights-5/optimizer_step/.ATTRIBUTES/VARIABLE_VALUE

?0
fd
VARIABLE_VALUEquant_dense/kernel_min:layer_with_weights-5/kernel_min/.ATTRIBUTES/VARIABLE_VALUE
fd
VARIABLE_VALUEquant_dense/kernel_max:layer_with_weights-5/kernel_max/.ATTRIBUTES/VARIABLE_VALUE
 
xv
VARIABLE_VALUEquant_dense/post_activation_minClayer_with_weights-5/post_activation_min/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEquant_dense/post_activation_maxClayer_with_weights-5/post_activation_max/.ATTRIBUTES/VARIABLE_VALUE
 
1
Q0
R1
>2
@3
A4
C5
D6

Q0
R1
 
?
?layer_metrics
?metrics
F	variables
?layers
 ?layer_regularization_losses
Gtrainable_variables
Hregularization_losses
?non_trainable_variables
GE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEconv2d/bias&variables/3/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEconv2d/kernel&variables/4/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUE
dense/bias'variables/13/.ATTRIBUTES/VARIABLE_VALUE
IG
VARIABLE_VALUEdense/kernel'variables/14/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1
*
0
1
2
3
4
5
 
v
0
1
2
3
4
5
6
7
#8
,9
510
>11
@12
A13
C14
D15
 
 
 
 

0
1
2

O0

O0
 
?
?layer_metrics
?metrics
]	variables
?layers
 ?layer_regularization_losses
^trainable_variables
_regularization_losses
?non_trainable_variables

P0
?2
 
 

0
 
#
0
1
2
3
4
 
 
 
?
?layer_metrics
?metrics
g	variables
?layers
 ?layer_regularization_losses
htrainable_variables
iregularization_losses
?non_trainable_variables
 
 

"0
 

#0
 
 
 
?
?layer_metrics
?metrics
p	variables
?layers
 ?layer_regularization_losses
qtrainable_variables
rregularization_losses
?non_trainable_variables
 
 

+0
 

,0
 
 
 
?
?layer_metrics
?metrics
y	variables
?layers
 ?layer_regularization_losses
ztrainable_variables
{regularization_losses
?non_trainable_variables
 
 

40
 

50

Q0

Q0
 
?
?layer_metrics
?metrics
?	variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?non_trainable_variables

R0
?2
 
 

=0
 
#
>0
@1
A2
C3
D4
8

?total

?count
?	variables
?	keras_api
I

?total

?count
?
_fn_kwargs
?	variables
?	keras_api
 
 
 
 
 

min_var
max_var
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 

@min_var
Amax_var
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
jh
VARIABLE_VALUEAdam/conv2d/bias/mBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d/kernel/mBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/dense/bias/mCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense/kernel/mCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/conv2d/bias/vBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/conv2d/kernel/vBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUEAdam/dense/bias/vCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUEAdam/dense/kernel/vCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????(1*
dtype0*$
shape:?????????(1
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1!quantize_layer/quantize_layer_min!quantize_layer/quantize_layer_maxconv2d/kernelquant_conv2d/kernel_minquant_conv2d/kernel_maxconv2d/bias quant_conv2d/post_activation_min quant_conv2d/post_activation_maxdense/kernelquant_dense/kernel_minquant_dense/kernel_max
dense/biasquant_dense/post_activation_minquant_dense/post_activation_max*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_387819
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename5quantize_layer/quantize_layer_min/Read/ReadVariableOp5quantize_layer/quantize_layer_max/Read/ReadVariableOp1quantize_layer/optimizer_step/Read/ReadVariableOp/quant_conv2d/optimizer_step/Read/ReadVariableOp+quant_conv2d/kernel_min/Read/ReadVariableOp+quant_conv2d/kernel_max/Read/ReadVariableOp4quant_conv2d/post_activation_min/Read/ReadVariableOp4quant_conv2d/post_activation_max/Read/ReadVariableOp6quant_max_pooling2d/optimizer_step/Read/ReadVariableOp0quant_dropout/optimizer_step/Read/ReadVariableOp0quant_flatten/optimizer_step/Read/ReadVariableOp.quant_dense/optimizer_step/Read/ReadVariableOp*quant_dense/kernel_min/Read/ReadVariableOp*quant_dense/kernel_max/Read/ReadVariableOp3quant_dense/post_activation_min/Read/ReadVariableOp3quant_dense/post_activation_max/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOpConst*2
Tin+
)2'	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__traced_save_388525
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename!quantize_layer/quantize_layer_min!quantize_layer/quantize_layer_maxquantize_layer/optimizer_stepquant_conv2d/optimizer_stepquant_conv2d/kernel_minquant_conv2d/kernel_max quant_conv2d/post_activation_min quant_conv2d/post_activation_max"quant_max_pooling2d/optimizer_stepquant_dropout/optimizer_stepquant_flatten/optimizer_stepquant_dense/optimizer_stepquant_dense/kernel_minquant_dense/kernel_maxquant_dense/post_activation_minquant_dense/post_activation_maxbeta_1beta_2decaylearning_rate	Adam/iterconv2d/biasconv2d/kernel
dense/biasdense/kerneltotalcounttotal_1count_1Adam/conv2d/bias/mAdam/conv2d/kernel/mAdam/dense/bias/mAdam/dense/kernel/mAdam/conv2d/bias/vAdam/conv2d/kernel/vAdam/dense/bias/vAdam/dense/kernel/v*1
Tin*
(2&*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference__traced_restore_388646??
?]
?	
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_388212

inputsI
/lastvaluequant_batchmin_readvariableop_resource:
3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?LastValueQuant/AssignMaxLast?LastValueQuant/AssignMinLast?&LastValueQuant/BatchMax/ReadVariableOp?&LastValueQuant/BatchMin/ReadVariableOp??LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1?ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2?2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp?-MovingAvgQuantize/AssignMaxEma/ReadVariableOp?2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp?-MovingAvgQuantize/AssignMinEma/ReadVariableOp?8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:
*
dtype02(
&LastValueQuant/BatchMin/ReadVariableOp?
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMin/reduction_indices?
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMin?
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:
*
dtype02(
&LastValueQuant/BatchMax/ReadVariableOp?
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMax/reduction_indices?
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMaxy
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
LastValueQuant/truediv/y?
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/truediv?
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:2
LastValueQuant/Minimumq
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
LastValueQuant/mul/y?
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/mul?
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:2
LastValueQuant/Maximum?
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMinLast?
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMaxLast?
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:
*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2?
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:
*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel?
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const?
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin?
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const_1?
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y?
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y?
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum?
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2&
$MovingAvgQuantize/AssignMinEma/decay?
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOp?
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/sub?
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mul?
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp?
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2&
$MovingAvgQuantize/AssignMaxEma/decay?
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp?
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/sub?
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mul?
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp?
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????2+
)MovingAvgQuantize/FakeQuantWithMinMaxVars?
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????(1: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2?
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:W S
/
_output_shapes
:?????????(1
 
_user_specified_nameinputs
?b
?
G__inference_quant_dense_layer_call_and_return_conditional_losses_388391

inputs>
+lastvaluequant_rank_readvariableop_resource:	?/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?LastValueQuant/AssignMaxLast?LastValueQuant/AssignMinLast?&LastValueQuant/BatchMax/ReadVariableOp?&LastValueQuant/BatchMin/ReadVariableOp?5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp?7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1?7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2?2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp?-MovingAvgQuantize/AssignMaxEma/ReadVariableOp?2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp?-MovingAvgQuantize/AssignMinEma/ReadVariableOp?8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"LastValueQuant/Rank/ReadVariableOpl
LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :2
LastValueQuant/Rankz
LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
LastValueQuant/range/startz
LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
LastValueQuant/range/delta?
LastValueQuant/rangeRange#LastValueQuant/range/start:output:0LastValueQuant/Rank:output:0#LastValueQuant/range/delta:output:0*
_output_shapes
:2
LastValueQuant/range?
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&LastValueQuant/BatchMin/ReadVariableOp?
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: 2
LastValueQuant/BatchMin?
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes
:	?*
dtype02&
$LastValueQuant/Rank_1/ReadVariableOpp
LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2
LastValueQuant/Rank_1~
LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2
LastValueQuant/range_1/start~
LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
LastValueQuant/range_1/delta?
LastValueQuant/range_1Range%LastValueQuant/range_1/start:output:0LastValueQuant/Rank_1:output:0%LastValueQuant/range_1/delta:output:0*
_output_shapes
:2
LastValueQuant/range_1?
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&LastValueQuant/BatchMax/ReadVariableOp?
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:0LastValueQuant/range_1:output:0*
T0*
_output_shapes
: 2
LastValueQuant/BatchMaxy
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
LastValueQuant/truediv/y?
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: 2
LastValueQuant/truediv?
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: 2
LastValueQuant/Minimumq
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
LastValueQuant/mul/y?
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: 2
LastValueQuant/mul?
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: 2
LastValueQuant/Maximum?
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMinLast?
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMaxLast?
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes
:	?*
dtype027
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp?
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype029
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1?
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype029
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2?
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	?*
narrow_range(2(
&LastValueQuant/FakeQuantWithMinMaxVars?
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
MovingAvgQuantize/Const?
MovingAvgQuantize/BatchMinMinBiasAdd:output:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin?
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
MovingAvgQuantize/Const_1?
MovingAvgQuantize/BatchMaxMaxBiasAdd:output:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y?
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y?
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum?
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2&
$MovingAvgQuantize/AssignMinEma/decay?
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOp?
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/sub?
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mul?
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp?
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2&
$MovingAvgQuantize/AssignMaxEma/decay?
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp?
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/sub?
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mul?
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp?
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:?????????2+
)MovingAvgQuantize/FakeQuantWithMinMaxVars?
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
,__inference_quant_dense_layer_call_fn_388298

inputs
unknown:	?
	unknown_0: 
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_quant_dense_layer_call_and_return_conditional_losses_3872482
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
e
I__inference_quant_flatten_layer_call_and_return_conditional_losses_387399

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
? 
?
P__inference_ExtAudioDataModelQAT_layer_call_and_return_conditional_losses_387740
input_1
quantize_layer_387705: 
quantize_layer_387707: -
quant_conv2d_387710:
!
quant_conv2d_387712:!
quant_conv2d_387714:!
quant_conv2d_387716:
quant_conv2d_387718: 
quant_conv2d_387720: %
quant_dense_387726:	?
quant_dense_387728: 
quant_dense_387730:  
quant_dense_387732:
quant_dense_387734: 
quant_dense_387736: 
identity??$quant_conv2d/StatefulPartitionedCall?#quant_dense/StatefulPartitionedCall?&quantize_layer/StatefulPartitionedCall?
&quantize_layer/StatefulPartitionedCallStatefulPartitionedCallinput_1quantize_layer_387705quantize_layer_387707*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_quantize_layer_layer_call_and_return_conditional_losses_3871652(
&quantize_layer/StatefulPartitionedCall?
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0quant_conv2d_387710quant_conv2d_387712quant_conv2d_387714quant_conv2d_387716quant_conv2d_387718quant_conv2d_387720*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_3871922&
$quant_conv2d/StatefulPartitionedCall?
#quant_max_pooling2d/PartitionedCallPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_quant_max_pooling2d_layer_call_and_return_conditional_losses_3872112%
#quant_max_pooling2d/PartitionedCall?
quant_dropout/PartitionedCallPartitionedCall,quant_max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_quant_dropout_layer_call_and_return_conditional_losses_3872182
quant_dropout/PartitionedCall?
quant_flatten/PartitionedCallPartitionedCall&quant_dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_quant_flatten_layer_call_and_return_conditional_losses_3872262
quant_flatten/PartitionedCall?
#quant_dense/StatefulPartitionedCallStatefulPartitionedCall&quant_flatten/PartitionedCall:output:0quant_dense_387726quant_dense_387728quant_dense_387730quant_dense_387732quant_dense_387734quant_dense_387736*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_quant_dense_layer_call_and_return_conditional_losses_3872482%
#quant_dense/StatefulPartitionedCall?
IdentityIdentity,quant_dense/StatefulPartitionedCall:output:0%^quant_conv2d/StatefulPartitionedCall$^quant_dense/StatefulPartitionedCall'^quantize_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????(1: : : : : : : : : : : : : : 2L
$quant_conv2d/StatefulPartitionedCall$quant_conv2d/StatefulPartitionedCall2J
#quant_dense/StatefulPartitionedCall#quant_dense/StatefulPartitionedCall2P
&quantize_layer/StatefulPartitionedCall&quantize_layer/StatefulPartitionedCall:X T
/
_output_shapes
:?????????(1
!
_user_specified_name	input_1
?
P
4__inference_quant_max_pooling2d_layer_call_fn_388222

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_quant_max_pooling2d_layer_call_and_return_conditional_losses_3874382
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
J
.__inference_quant_flatten_layer_call_fn_388269

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_quant_flatten_layer_call_and_return_conditional_losses_3873992
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?&
?
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_387192

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:
X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity??BiasAdd/ReadVariableOp??LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1?ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2?8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:
*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2?
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:
*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel?
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????2+
)MovingAvgQuantize/FakeQuantWithMinMaxVars?
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????(1: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2?
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:W S
/
_output_shapes
:?????????(1
 
_user_specified_nameinputs
? 
?
P__inference_ExtAudioDataModelQAT_layer_call_and_return_conditional_losses_387263

inputs
quantize_layer_387166: 
quantize_layer_387168: -
quant_conv2d_387193:
!
quant_conv2d_387195:!
quant_conv2d_387197:!
quant_conv2d_387199:
quant_conv2d_387201: 
quant_conv2d_387203: %
quant_dense_387249:	?
quant_dense_387251: 
quant_dense_387253:  
quant_dense_387255:
quant_dense_387257: 
quant_dense_387259: 
identity??$quant_conv2d/StatefulPartitionedCall?#quant_dense/StatefulPartitionedCall?&quantize_layer/StatefulPartitionedCall?
&quantize_layer/StatefulPartitionedCallStatefulPartitionedCallinputsquantize_layer_387166quantize_layer_387168*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_quantize_layer_layer_call_and_return_conditional_losses_3871652(
&quantize_layer/StatefulPartitionedCall?
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0quant_conv2d_387193quant_conv2d_387195quant_conv2d_387197quant_conv2d_387199quant_conv2d_387201quant_conv2d_387203*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_3871922&
$quant_conv2d/StatefulPartitionedCall?
#quant_max_pooling2d/PartitionedCallPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_quant_max_pooling2d_layer_call_and_return_conditional_losses_3872112%
#quant_max_pooling2d/PartitionedCall?
quant_dropout/PartitionedCallPartitionedCall,quant_max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_quant_dropout_layer_call_and_return_conditional_losses_3872182
quant_dropout/PartitionedCall?
quant_flatten/PartitionedCallPartitionedCall&quant_dropout/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_quant_flatten_layer_call_and_return_conditional_losses_3872262
quant_flatten/PartitionedCall?
#quant_dense/StatefulPartitionedCallStatefulPartitionedCall&quant_flatten/PartitionedCall:output:0quant_dense_387249quant_dense_387251quant_dense_387253quant_dense_387255quant_dense_387257quant_dense_387259*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_quant_dense_layer_call_and_return_conditional_losses_3872482%
#quant_dense/StatefulPartitionedCall?
IdentityIdentity,quant_dense/StatefulPartitionedCall:output:0%^quant_conv2d/StatefulPartitionedCall$^quant_dense/StatefulPartitionedCall'^quantize_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????(1: : : : : : : : : : : : : : 2L
$quant_conv2d/StatefulPartitionedCall$quant_conv2d/StatefulPartitionedCall2J
#quant_dense/StatefulPartitionedCall#quant_dense/StatefulPartitionedCall2P
&quantize_layer/StatefulPartitionedCall&quantize_layer/StatefulPartitionedCall:W S
/
_output_shapes
:?????????(1
 
_user_specified_nameinputs
??
?
!__inference__wrapped_model_387137
input_1o
eextaudiodatamodelqat_quantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: q
gextaudiodatamodelqat_quantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: ?
jextaudiodatamodelqat_quant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:
z
lextaudiodatamodelqat_quant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:z
lextaudiodatamodelqat_quant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:O
Aextaudiodatamodelqat_quant_conv2d_biasadd_readvariableop_resource:m
cextaudiodatamodelqat_quant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: o
eextaudiodatamodelqat_quant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: r
_extaudiodatamodelqat_quant_dense_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:	?k
aextaudiodatamodelqat_quant_dense_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: k
aextaudiodatamodelqat_quant_dense_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: N
@extaudiodatamodelqat_quant_dense_biasadd_readvariableop_resource:l
bextaudiodatamodelqat_quant_dense_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: n
dextaudiodatamodelqat_quant_dense_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity??8ExtAudioDataModelQAT/quant_conv2d/BiasAdd/ReadVariableOp?aExtAudioDataModelQAT/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?cExtAudioDataModelQAT/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1?cExtAudioDataModelQAT/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2?ZExtAudioDataModelQAT/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?\ExtAudioDataModelQAT/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?7ExtAudioDataModelQAT/quant_dense/BiasAdd/ReadVariableOp?VExtAudioDataModelQAT/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp?XExtAudioDataModelQAT/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1?XExtAudioDataModelQAT/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2?YExtAudioDataModelQAT/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?[ExtAudioDataModelQAT/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?\ExtAudioDataModelQAT/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?^ExtAudioDataModelQAT/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
\ExtAudioDataModelQAT/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpeextaudiodatamodelqat_quantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02^
\ExtAudioDataModelQAT/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
^ExtAudioDataModelQAT/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpgextaudiodatamodelqat_quantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02`
^ExtAudioDataModelQAT/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
MExtAudioDataModelQAT/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinput_1dExtAudioDataModelQAT/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0fExtAudioDataModelQAT/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????(12O
MExtAudioDataModelQAT/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars?
aExtAudioDataModelQAT/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpjextaudiodatamodelqat_quant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:
*
dtype02c
aExtAudioDataModelQAT/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?
cExtAudioDataModelQAT/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOplextaudiodatamodelqat_quant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02e
cExtAudioDataModelQAT/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1?
cExtAudioDataModelQAT/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOplextaudiodatamodelqat_quant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02e
cExtAudioDataModelQAT/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2?
RExtAudioDataModelQAT/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChanneliExtAudioDataModelQAT/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0kExtAudioDataModelQAT/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0kExtAudioDataModelQAT/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:
*
narrow_range(2T
RExtAudioDataModelQAT/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel?
(ExtAudioDataModelQAT/quant_conv2d/Conv2DConv2DWExtAudioDataModelQAT/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0\ExtAudioDataModelQAT/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2*
(ExtAudioDataModelQAT/quant_conv2d/Conv2D?
8ExtAudioDataModelQAT/quant_conv2d/BiasAdd/ReadVariableOpReadVariableOpAextaudiodatamodelqat_quant_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02:
8ExtAudioDataModelQAT/quant_conv2d/BiasAdd/ReadVariableOp?
)ExtAudioDataModelQAT/quant_conv2d/BiasAddBiasAdd1ExtAudioDataModelQAT/quant_conv2d/Conv2D:output:0@ExtAudioDataModelQAT/quant_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2+
)ExtAudioDataModelQAT/quant_conv2d/BiasAdd?
&ExtAudioDataModelQAT/quant_conv2d/ReluRelu2ExtAudioDataModelQAT/quant_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2(
&ExtAudioDataModelQAT/quant_conv2d/Relu?
ZExtAudioDataModelQAT/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpcextaudiodatamodelqat_quant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02\
ZExtAudioDataModelQAT/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
\ExtAudioDataModelQAT/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpeextaudiodatamodelqat_quant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02^
\ExtAudioDataModelQAT/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
KExtAudioDataModelQAT/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars4ExtAudioDataModelQAT/quant_conv2d/Relu:activations:0bExtAudioDataModelQAT/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0dExtAudioDataModelQAT/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????2M
KExtAudioDataModelQAT/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars?
0ExtAudioDataModelQAT/quant_max_pooling2d/MaxPoolMaxPoolUExtAudioDataModelQAT/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*/
_output_shapes
:?????????
*
ksize
*
paddingVALID*
strides
22
0ExtAudioDataModelQAT/quant_max_pooling2d/MaxPool?
+ExtAudioDataModelQAT/quant_dropout/IdentityIdentity9ExtAudioDataModelQAT/quant_max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:?????????
2-
+ExtAudioDataModelQAT/quant_dropout/Identity?
(ExtAudioDataModelQAT/quant_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2*
(ExtAudioDataModelQAT/quant_flatten/Const?
*ExtAudioDataModelQAT/quant_flatten/ReshapeReshape4ExtAudioDataModelQAT/quant_dropout/Identity:output:01ExtAudioDataModelQAT/quant_flatten/Const:output:0*
T0*(
_output_shapes
:??????????2,
*ExtAudioDataModelQAT/quant_flatten/Reshape?
VExtAudioDataModelQAT/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp_extaudiodatamodelqat_quant_dense_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
:	?*
dtype02X
VExtAudioDataModelQAT/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp?
XExtAudioDataModelQAT/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpaextaudiodatamodelqat_quant_dense_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02Z
XExtAudioDataModelQAT/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1?
XExtAudioDataModelQAT/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpaextaudiodatamodelqat_quant_dense_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype02Z
XExtAudioDataModelQAT/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2?
GExtAudioDataModelQAT/quant_dense/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars^ExtAudioDataModelQAT/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0`ExtAudioDataModelQAT/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0`ExtAudioDataModelQAT/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	?*
narrow_range(2I
GExtAudioDataModelQAT/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars?
'ExtAudioDataModelQAT/quant_dense/MatMulMatMul3ExtAudioDataModelQAT/quant_flatten/Reshape:output:0QExtAudioDataModelQAT/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:?????????2)
'ExtAudioDataModelQAT/quant_dense/MatMul?
7ExtAudioDataModelQAT/quant_dense/BiasAdd/ReadVariableOpReadVariableOp@extaudiodatamodelqat_quant_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype029
7ExtAudioDataModelQAT/quant_dense/BiasAdd/ReadVariableOp?
(ExtAudioDataModelQAT/quant_dense/BiasAddBiasAdd1ExtAudioDataModelQAT/quant_dense/MatMul:product:0?ExtAudioDataModelQAT/quant_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2*
(ExtAudioDataModelQAT/quant_dense/BiasAdd?
YExtAudioDataModelQAT/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpbextaudiodatamodelqat_quant_dense_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02[
YExtAudioDataModelQAT/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
[ExtAudioDataModelQAT/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpdextaudiodatamodelqat_quant_dense_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02]
[ExtAudioDataModelQAT/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
JExtAudioDataModelQAT/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars1ExtAudioDataModelQAT/quant_dense/BiasAdd:output:0aExtAudioDataModelQAT/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0cExtAudioDataModelQAT/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:?????????2L
JExtAudioDataModelQAT/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars?
IdentityIdentityTExtAudioDataModelQAT/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:09^ExtAudioDataModelQAT/quant_conv2d/BiasAdd/ReadVariableOpb^ExtAudioDataModelQAT/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpd^ExtAudioDataModelQAT/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1d^ExtAudioDataModelQAT/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2[^ExtAudioDataModelQAT/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp]^ExtAudioDataModelQAT/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_18^ExtAudioDataModelQAT/quant_dense/BiasAdd/ReadVariableOpW^ExtAudioDataModelQAT/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpY^ExtAudioDataModelQAT/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Y^ExtAudioDataModelQAT/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Z^ExtAudioDataModelQAT/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp\^ExtAudioDataModelQAT/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1]^ExtAudioDataModelQAT/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_^ExtAudioDataModelQAT/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????(1: : : : : : : : : : : : : : 2t
8ExtAudioDataModelQAT/quant_conv2d/BiasAdd/ReadVariableOp8ExtAudioDataModelQAT/quant_conv2d/BiasAdd/ReadVariableOp2?
aExtAudioDataModelQAT/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpaExtAudioDataModelQAT/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2?
cExtAudioDataModelQAT/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1cExtAudioDataModelQAT/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12?
cExtAudioDataModelQAT/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2cExtAudioDataModelQAT/quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22?
ZExtAudioDataModelQAT/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpZExtAudioDataModelQAT/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2?
\ExtAudioDataModelQAT/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1\ExtAudioDataModelQAT/quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7ExtAudioDataModelQAT/quant_dense/BiasAdd/ReadVariableOp7ExtAudioDataModelQAT/quant_dense/BiasAdd/ReadVariableOp2?
VExtAudioDataModelQAT/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpVExtAudioDataModelQAT/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2?
XExtAudioDataModelQAT/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1XExtAudioDataModelQAT/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12?
XExtAudioDataModelQAT/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2XExtAudioDataModelQAT/quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22?
YExtAudioDataModelQAT/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpYExtAudioDataModelQAT/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2?
[ExtAudioDataModelQAT/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1[ExtAudioDataModelQAT/quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12?
\ExtAudioDataModelQAT/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp\ExtAudioDataModelQAT/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2?
^ExtAudioDataModelQAT/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1^ExtAudioDataModelQAT/quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:X T
/
_output_shapes
:?????????(1
!
_user_specified_name	input_1
?"
?
G__inference_quant_dense_layer_call_and_return_conditional_losses_387248

inputsQ
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:	?J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity??BiasAdd/ReadVariableOp?5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp?7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1?7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2?8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
:	?*
dtype027
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp?
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype029
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1?
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype029
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2?
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	?*
narrow_range(2(
&LastValueQuant/FakeQuantWithMinMaxVars?
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:?????????2+
)MovingAvgQuantize/FakeQuantWithMinMaxVars?
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
k
O__inference_quant_max_pooling2d_layer_call_and_return_conditional_losses_388227

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????
*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
5__inference_ExtAudioDataModelQAT_layer_call_fn_387885

inputs
unknown: 
	unknown_0: #
	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5: 
	unknown_6: 
	unknown_7:	?
	unknown_8: 
	unknown_9: 

unknown_10:

unknown_11: 

unknown_12: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_ExtAudioDataModelQAT_layer_call_and_return_conditional_losses_3876382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????(1: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????(1
 
_user_specified_nameinputs
?
g
.__inference_quant_dropout_layer_call_fn_388242

inputs
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_quant_dropout_layer_call_and_return_conditional_losses_3874222
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?
g
I__inference_quant_dropout_layer_call_and_return_conditional_losses_388247

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????
2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????
2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
5__inference_ExtAudioDataModelQAT_layer_call_fn_387294
input_1
unknown: 
	unknown_0: #
	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5: 
	unknown_6: 
	unknown_7:	?
	unknown_8: 
	unknown_9: 

unknown_10:

unknown_11: 

unknown_12: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_ExtAudioDataModelQAT_layer_call_and_return_conditional_losses_3872632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????(1: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????(1
!
_user_specified_name	input_1
?
?
-__inference_quant_conv2d_layer_call_fn_388142

inputs!
unknown:

	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_3875102
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????(1: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????(1
 
_user_specified_nameinputs
??
?
"__inference__traced_restore_388646
file_prefix<
2assignvariableop_quantize_layer_quantize_layer_min: >
4assignvariableop_1_quantize_layer_quantize_layer_max: :
0assignvariableop_2_quantize_layer_optimizer_step: 8
.assignvariableop_3_quant_conv2d_optimizer_step: 8
*assignvariableop_4_quant_conv2d_kernel_min:8
*assignvariableop_5_quant_conv2d_kernel_max:=
3assignvariableop_6_quant_conv2d_post_activation_min: =
3assignvariableop_7_quant_conv2d_post_activation_max: ?
5assignvariableop_8_quant_max_pooling2d_optimizer_step: 9
/assignvariableop_9_quant_dropout_optimizer_step: :
0assignvariableop_10_quant_flatten_optimizer_step: 8
.assignvariableop_11_quant_dense_optimizer_step: 4
*assignvariableop_12_quant_dense_kernel_min: 4
*assignvariableop_13_quant_dense_kernel_max: =
3assignvariableop_14_quant_dense_post_activation_min: =
3assignvariableop_15_quant_dense_post_activation_max: $
assignvariableop_16_beta_1: $
assignvariableop_17_beta_2: #
assignvariableop_18_decay: +
!assignvariableop_19_learning_rate: '
assignvariableop_20_adam_iter:	 -
assignvariableop_21_conv2d_bias:;
!assignvariableop_22_conv2d_kernel:
,
assignvariableop_23_dense_bias:3
 assignvariableop_24_dense_kernel:	?#
assignvariableop_25_total: #
assignvariableop_26_count: %
assignvariableop_27_total_1: %
assignvariableop_28_count_1: 4
&assignvariableop_29_adam_conv2d_bias_m:B
(assignvariableop_30_adam_conv2d_kernel_m:
3
%assignvariableop_31_adam_dense_bias_m::
'assignvariableop_32_adam_dense_kernel_m:	?4
&assignvariableop_33_adam_conv2d_bias_v:B
(assignvariableop_34_adam_conv2d_kernel_v:
3
%assignvariableop_35_adam_dense_bias_v::
'assignvariableop_36_adam_dense_kernel_v:	?
identity_38??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*?
value?B?&BBlayer_with_weights-0/quantize_layer_min/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/quantize_layer_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-5/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-5/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-5/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::*4
dtypes*
(2&	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp2assignvariableop_quantize_layer_quantize_layer_minIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp4assignvariableop_1_quantize_layer_quantize_layer_maxIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp0assignvariableop_2_quantize_layer_optimizer_stepIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp.assignvariableop_3_quant_conv2d_optimizer_stepIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp*assignvariableop_4_quant_conv2d_kernel_minIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp*assignvariableop_5_quant_conv2d_kernel_maxIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp3assignvariableop_6_quant_conv2d_post_activation_minIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp3assignvariableop_7_quant_conv2d_post_activation_maxIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp5assignvariableop_8_quant_max_pooling2d_optimizer_stepIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp/assignvariableop_9_quant_dropout_optimizer_stepIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp0assignvariableop_10_quant_flatten_optimizer_stepIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp.assignvariableop_11_quant_dense_optimizer_stepIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp*assignvariableop_12_quant_dense_kernel_minIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp*assignvariableop_13_quant_dense_kernel_maxIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp3assignvariableop_14_quant_dense_post_activation_minIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp3assignvariableop_15_quant_dense_post_activation_maxIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_beta_1Identity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_beta_2Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_decayIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp!assignvariableop_19_learning_rateIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_adam_iterIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_conv2d_biasIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp!assignvariableop_22_conv2d_kernelIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_dense_biasIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp assignvariableop_24_dense_kernelIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpassignvariableop_25_totalIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpassignvariableop_26_countIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOpassignvariableop_27_total_1Identity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOpassignvariableop_28_count_1Identity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp&assignvariableop_29_adam_conv2d_bias_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp(assignvariableop_30_adam_conv2d_kernel_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp%assignvariableop_31_adam_dense_bias_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp'assignvariableop_32_adam_dense_kernel_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp&assignvariableop_33_adam_conv2d_bias_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp(assignvariableop_34_adam_conv2d_kernel_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp%assignvariableop_35_adam_dense_bias_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp'assignvariableop_36_adam_dense_kernel_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_369
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_37Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_37?
Identity_38IdentityIdentity_37:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_38"#
identity_38Identity_38:output:0*_
_input_shapesN
L: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?]
?	
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_387510

inputsI
/lastvaluequant_batchmin_readvariableop_resource:
3
%lastvaluequant_assignminlast_resource:3
%lastvaluequant_assignmaxlast_resource:-
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?LastValueQuant/AssignMaxLast?LastValueQuant/AssignMinLast?&LastValueQuant/BatchMax/ReadVariableOp?&LastValueQuant/BatchMin/ReadVariableOp??LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1?ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2?2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp?-MovingAvgQuantize/AssignMaxEma/ReadVariableOp?2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp?-MovingAvgQuantize/AssignMinEma/ReadVariableOp?8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:
*
dtype02(
&LastValueQuant/BatchMin/ReadVariableOp?
)LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMin/reduction_indices?
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:02LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMin?
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:
*
dtype02(
&LastValueQuant/BatchMax/ReadVariableOp?
)LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2+
)LastValueQuant/BatchMax/reduction_indices?
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:02LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:2
LastValueQuant/BatchMaxy
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
LastValueQuant/truediv/y?
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/truediv?
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
:2
LastValueQuant/Minimumq
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
LastValueQuant/mul/y?
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:2
LastValueQuant/mul?
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
:2
LastValueQuant/Maximum?
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMinLast?
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMaxLast?
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp/lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:
*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2?
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:
*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel?
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const?
MovingAvgQuantize/BatchMinMinRelu:activations:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin?
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
MovingAvgQuantize/Const_1?
MovingAvgQuantize/BatchMaxMaxRelu:activations:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y?
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y?
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum?
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2&
$MovingAvgQuantize/AssignMinEma/decay?
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOp?
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/sub?
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mul?
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp?
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2&
$MovingAvgQuantize/AssignMaxEma/decay?
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp?
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/sub?
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mul?
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp?
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????2+
)MovingAvgQuantize/FakeQuantWithMinMaxVars?
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????(1: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2?
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:W S
/
_output_shapes
:?????????(1
 
_user_specified_nameinputs
?
?
,__inference_quant_dense_layer_call_fn_388315

inputs
unknown:	?
	unknown_0: 
	unknown_1: 
	unknown_2:
	unknown_3: 
	unknown_4: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_quant_dense_layer_call_and_return_conditional_losses_3873702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
J__inference_quantize_layer_layer_call_and_return_conditional_losses_388087

inputsK
Aallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: M
Callvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity??8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCallvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????(12+
)AllValuesQuantize/FakeQuantWithMinMaxVars?
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:09^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*/
_output_shapes
:?????????(12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(1: : 2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:W S
/
_output_shapes
:?????????(1
 
_user_specified_nameinputs
?
J
.__inference_quant_flatten_layer_call_fn_388264

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_quant_flatten_layer_call_and_return_conditional_losses_3872262
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
/__inference_quantize_layer_layer_call_fn_388069

inputs
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(1*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_quantize_layer_layer_call_and_return_conditional_losses_3871652
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????(12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(1: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????(1
 
_user_specified_nameinputs
?
?
J__inference_quantize_layer_layer_call_and_return_conditional_losses_387165

inputsK
Aallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: M
Callvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity??8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAallvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCallvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????(12+
)AllValuesQuantize/FakeQuantWithMinMaxVars?
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:09^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*/
_output_shapes
:?????????(12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(1: : 2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:W S
/
_output_shapes
:?????????(1
 
_user_specified_nameinputs
?b
?
G__inference_quant_dense_layer_call_and_return_conditional_losses_387370

inputs>
+lastvaluequant_rank_readvariableop_resource:	?/
%lastvaluequant_assignminlast_resource: /
%lastvaluequant_assignmaxlast_resource: -
biasadd_readvariableop_resource:@
6movingavgquantize_assignminema_readvariableop_resource: @
6movingavgquantize_assignmaxema_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?LastValueQuant/AssignMaxLast?LastValueQuant/AssignMinLast?&LastValueQuant/BatchMax/ReadVariableOp?&LastValueQuant/BatchMin/ReadVariableOp?5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp?7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1?7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2?2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp?-MovingAvgQuantize/AssignMaxEma/ReadVariableOp?2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp?-MovingAvgQuantize/AssignMinEma/ReadVariableOp?8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
"LastValueQuant/Rank/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes
:	?*
dtype02$
"LastValueQuant/Rank/ReadVariableOpl
LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :2
LastValueQuant/Rankz
LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
LastValueQuant/range/startz
LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
LastValueQuant/range/delta?
LastValueQuant/rangeRange#LastValueQuant/range/start:output:0LastValueQuant/Rank:output:0#LastValueQuant/range/delta:output:0*
_output_shapes
:2
LastValueQuant/range?
&LastValueQuant/BatchMin/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&LastValueQuant/BatchMin/ReadVariableOp?
LastValueQuant/BatchMinMin.LastValueQuant/BatchMin/ReadVariableOp:value:0LastValueQuant/range:output:0*
T0*
_output_shapes
: 2
LastValueQuant/BatchMin?
$LastValueQuant/Rank_1/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes
:	?*
dtype02&
$LastValueQuant/Rank_1/ReadVariableOpp
LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2
LastValueQuant/Rank_1~
LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2
LastValueQuant/range_1/start~
LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
LastValueQuant/range_1/delta?
LastValueQuant/range_1Range%LastValueQuant/range_1/start:output:0LastValueQuant/Rank_1:output:0%LastValueQuant/range_1/delta:output:0*
_output_shapes
:2
LastValueQuant/range_1?
&LastValueQuant/BatchMax/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes
:	?*
dtype02(
&LastValueQuant/BatchMax/ReadVariableOp?
LastValueQuant/BatchMaxMax.LastValueQuant/BatchMax/ReadVariableOp:value:0LastValueQuant/range_1:output:0*
T0*
_output_shapes
: 2
LastValueQuant/BatchMaxy
LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
LastValueQuant/truediv/y?
LastValueQuant/truedivRealDiv LastValueQuant/BatchMax:output:0!LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: 2
LastValueQuant/truediv?
LastValueQuant/MinimumMinimum LastValueQuant/BatchMin:output:0LastValueQuant/truediv:z:0*
T0*
_output_shapes
: 2
LastValueQuant/Minimumq
LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
LastValueQuant/mul/y?
LastValueQuant/mulMul LastValueQuant/BatchMin:output:0LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: 2
LastValueQuant/mul?
LastValueQuant/MaximumMaximum LastValueQuant/BatchMax:output:0LastValueQuant/mul:z:0*
T0*
_output_shapes
: 2
LastValueQuant/Maximum?
LastValueQuant/AssignMinLastAssignVariableOp%lastvaluequant_assignminlast_resourceLastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMinLast?
LastValueQuant/AssignMaxLastAssignVariableOp%lastvaluequant_assignmaxlast_resourceLastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02
LastValueQuant/AssignMaxLast?
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp+lastvaluequant_rank_readvariableop_resource*
_output_shapes
:	?*
dtype027
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp?
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp%lastvaluequant_assignminlast_resource^LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype029
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1?
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp%lastvaluequant_assignmaxlast_resource^LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype029
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2?
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	?*
narrow_range(2(
&LastValueQuant/FakeQuantWithMinMaxVars?
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2
MovingAvgQuantize/Const?
MovingAvgQuantize/BatchMinMinBiasAdd:output:0 MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMin?
MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2
MovingAvgQuantize/Const_1?
MovingAvgQuantize/BatchMaxMaxBiasAdd:output:0"MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/BatchMax
MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Minimum/y?
MovingAvgQuantize/MinimumMinimum#MovingAvgQuantize/BatchMin:output:0$MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Minimum
MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
MovingAvgQuantize/Maximum/y?
MovingAvgQuantize/MaximumMaximum#MovingAvgQuantize/BatchMax:output:0$MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2
MovingAvgQuantize/Maximum?
$MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2&
$MovingAvgQuantize/AssignMinEma/decay?
-MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMinEma/ReadVariableOp?
"MovingAvgQuantize/AssignMinEma/subSub5MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/sub?
"MovingAvgQuantize/AssignMinEma/mulMul&MovingAvgQuantize/AssignMinEma/sub:z:0-MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMinEma/mul?
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignminema_readvariableop_resource&MovingAvgQuantize/AssignMinEma/mul:z:0.^MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp?
$MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o?:2&
$MovingAvgQuantize/AssignMaxEma/decay?
-MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02/
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp?
"MovingAvgQuantize/AssignMaxEma/subSub5MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/sub?
"MovingAvgQuantize/AssignMaxEma/mulMul&MovingAvgQuantize/AssignMaxEma/sub:z:0-MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 2$
"MovingAvgQuantize/AssignMaxEma/mul?
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOp6movingavgquantize_assignmaxema_readvariableop_resource&MovingAvgQuantize/AssignMaxEma/mul:z:0.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype024
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp?
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp6movingavgquantize_assignminema_readvariableop_resource3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp6movingavgquantize_assignmaxema_readvariableop_resource3^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:?????????2+
)MovingAvgQuantize/FakeQuantWithMinMaxVars?
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp^LastValueQuant/AssignMaxLast^LastValueQuant/AssignMinLast'^LastValueQuant/BatchMax/ReadVariableOp'^LastValueQuant/BatchMin/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_23^MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMaxEma/ReadVariableOp3^MovingAvgQuantize/AssignMinEma/AssignSubVariableOp.^MovingAvgQuantize/AssignMinEma/ReadVariableOp9^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2<
LastValueQuant/AssignMaxLastLastValueQuant/AssignMaxLast2<
LastValueQuant/AssignMinLastLastValueQuant/AssignMinLast2P
&LastValueQuant/BatchMax/ReadVariableOp&LastValueQuant/BatchMax/ReadVariableOp2P
&LastValueQuant/BatchMin/ReadVariableOp&LastValueQuant/BatchMin/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22h
2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMaxEma/ReadVariableOp-MovingAvgQuantize/AssignMaxEma/ReadVariableOp2h
2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2^
-MovingAvgQuantize/AssignMinEma/ReadVariableOp-MovingAvgQuantize/AssignMinEma/ReadVariableOp2t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
k
O__inference_quant_max_pooling2d_layer_call_and_return_conditional_losses_388232

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????
*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
$__inference_signature_wrapper_387819
input_1
unknown: 
	unknown_0: #
	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5: 
	unknown_6: 
	unknown_7:	?
	unknown_8: 
	unknown_9: 

unknown_10:

unknown_11: 

unknown_12: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_3871372
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????(1: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????(1
!
_user_specified_name	input_1
?
k
O__inference_quant_max_pooling2d_layer_call_and_return_conditional_losses_387438

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????
*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?g
?
P__inference_ExtAudioDataModelQAT_layer_call_and_return_conditional_losses_387931

inputsZ
Pquantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource: \
Rquantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource: o
Uquant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:
e
Wquant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:e
Wquant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource::
,quant_conv2d_biasadd_readvariableop_resource:X
Nquant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: Z
Pquant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: ]
Jquant_dense_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:	?V
Lquant_dense_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: V
Lquant_dense_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: 9
+quant_dense_biasadd_readvariableop_resource:W
Mquant_dense_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource: Y
Oquant_dense_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity??#quant_conv2d/BiasAdd/ReadVariableOp?Lquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1?Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2?Equant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?"quant_dense/BiasAdd/ReadVariableOp?Aquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp?Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1?Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2?Dquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?Fquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpPquantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02I
Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpRquantize_layer_allvaluesquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02K
Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
8quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputsOquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????(12:
8quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars?
Lquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpUquant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:
*
dtype02N
Lquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpWquant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02P
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1?
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpWquant_conv2d_lastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02P
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2?
=quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelTquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Vquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Vquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:
*
narrow_range(2?
=quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel?
quant_conv2d/Conv2DConv2DBquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Gquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
quant_conv2d/Conv2D?
#quant_conv2d/BiasAdd/ReadVariableOpReadVariableOp,quant_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#quant_conv2d/BiasAdd/ReadVariableOp?
quant_conv2d/BiasAddBiasAddquant_conv2d/Conv2D:output:0+quant_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
quant_conv2d/BiasAdd?
quant_conv2d/ReluReluquant_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
quant_conv2d/Relu?
Equant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpNquant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02G
Equant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpPquant_conv2d_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02I
Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
6quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_conv2d/Relu:activations:0Mquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Oquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????28
6quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars?
quant_max_pooling2d/MaxPoolMaxPool@quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*/
_output_shapes
:?????????
*
ksize
*
paddingVALID*
strides
2
quant_max_pooling2d/MaxPool?
quant_dropout/IdentityIdentity$quant_max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:?????????
2
quant_dropout/Identity{
quant_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
quant_flatten/Const?
quant_flatten/ReshapeReshapequant_dropout/Identity:output:0quant_flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
quant_flatten/Reshape?
Aquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpJquant_dense_lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
:	?*
dtype02C
Aquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp?
Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpLquant_dense_lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02E
Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1?
Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOpLquant_dense_lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype02E
Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2?
2quant_dense/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsIquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Kquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Kquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	?*
narrow_range(24
2quant_dense/LastValueQuant/FakeQuantWithMinMaxVars?
quant_dense/MatMulMatMulquant_flatten/Reshape:output:0<quant_dense/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:?????????2
quant_dense/MatMul?
"quant_dense/BiasAdd/ReadVariableOpReadVariableOp+quant_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"quant_dense/BiasAdd/ReadVariableOp?
quant_dense/BiasAddBiasAddquant_dense/MatMul:product:0*quant_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
quant_dense/BiasAdd?
Dquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpMquant_dense_movingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02F
Dquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
Fquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpOquant_dense_movingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02H
Fquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
5quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_dense/BiasAdd:output:0Lquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:?????????27
5quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars?
IdentityIdentity?quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0$^quant_conv2d/BiasAdd/ReadVariableOpM^quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpO^quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1O^quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2F^quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpH^quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1#^quant_dense/BiasAdd/ReadVariableOpB^quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpD^quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1D^quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2E^quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1H^quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????(1: : : : : : : : : : : : : : 2J
#quant_conv2d/BiasAdd/ReadVariableOp#quant_conv2d/BiasAdd/ReadVariableOp2?
Lquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpLquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2?
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12?
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22?
Equant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpEquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2?
Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12H
"quant_dense/BiasAdd/ReadVariableOp"quant_dense/BiasAdd/ReadVariableOp2?
Aquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpAquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2?
Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12?
Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22?
Dquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2?
Fquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12?
Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2?
Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:W S
/
_output_shapes
:?????????(1
 
_user_specified_nameinputs
?"
?
P__inference_ExtAudioDataModelQAT_layer_call_and_return_conditional_losses_387778
input_1
quantize_layer_387743: 
quantize_layer_387745: -
quant_conv2d_387748:
!
quant_conv2d_387750:!
quant_conv2d_387752:!
quant_conv2d_387754:
quant_conv2d_387756: 
quant_conv2d_387758: %
quant_dense_387764:	?
quant_dense_387766: 
quant_dense_387768:  
quant_dense_387770:
quant_dense_387772: 
quant_dense_387774: 
identity??$quant_conv2d/StatefulPartitionedCall?#quant_dense/StatefulPartitionedCall?%quant_dropout/StatefulPartitionedCall?&quantize_layer/StatefulPartitionedCall?
&quantize_layer/StatefulPartitionedCallStatefulPartitionedCallinput_1quantize_layer_387743quantize_layer_387745*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(1* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_quantize_layer_layer_call_and_return_conditional_losses_3875582(
&quantize_layer/StatefulPartitionedCall?
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0quant_conv2d_387748quant_conv2d_387750quant_conv2d_387752quant_conv2d_387754quant_conv2d_387756quant_conv2d_387758*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_3875102&
$quant_conv2d/StatefulPartitionedCall?
#quant_max_pooling2d/PartitionedCallPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_quant_max_pooling2d_layer_call_and_return_conditional_losses_3874382%
#quant_max_pooling2d/PartitionedCall?
%quant_dropout/StatefulPartitionedCallStatefulPartitionedCall,quant_max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_quant_dropout_layer_call_and_return_conditional_losses_3874222'
%quant_dropout/StatefulPartitionedCall?
quant_flatten/PartitionedCallPartitionedCall.quant_dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_quant_flatten_layer_call_and_return_conditional_losses_3873992
quant_flatten/PartitionedCall?
#quant_dense/StatefulPartitionedCallStatefulPartitionedCall&quant_flatten/PartitionedCall:output:0quant_dense_387764quant_dense_387766quant_dense_387768quant_dense_387770quant_dense_387772quant_dense_387774*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_quant_dense_layer_call_and_return_conditional_losses_3873702%
#quant_dense/StatefulPartitionedCall?
IdentityIdentity,quant_dense/StatefulPartitionedCall:output:0%^quant_conv2d/StatefulPartitionedCall$^quant_dense/StatefulPartitionedCall&^quant_dropout/StatefulPartitionedCall'^quantize_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????(1: : : : : : : : : : : : : : 2L
$quant_conv2d/StatefulPartitionedCall$quant_conv2d/StatefulPartitionedCall2J
#quant_dense/StatefulPartitionedCall#quant_dense/StatefulPartitionedCall2N
%quant_dropout/StatefulPartitionedCall%quant_dropout/StatefulPartitionedCall2P
&quantize_layer/StatefulPartitionedCall&quantize_layer/StatefulPartitionedCall:X T
/
_output_shapes
:?????????(1
!
_user_specified_name	input_1
?
?
5__inference_ExtAudioDataModelQAT_layer_call_fn_387702
input_1
unknown: 
	unknown_0: #
	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5: 
	unknown_6: 
	unknown_7:	?
	unknown_8: 
	unknown_9: 

unknown_10:

unknown_11: 

unknown_12: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*&
_read_only_resource_inputs
	*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_ExtAudioDataModelQAT_layer_call_and_return_conditional_losses_3876382
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????(1: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????(1
!
_user_specified_name	input_1
?
e
I__inference_quant_flatten_layer_call_and_return_conditional_losses_387226

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?"
?
P__inference_ExtAudioDataModelQAT_layer_call_and_return_conditional_losses_387638

inputs
quantize_layer_387603: 
quantize_layer_387605: -
quant_conv2d_387608:
!
quant_conv2d_387610:!
quant_conv2d_387612:!
quant_conv2d_387614:
quant_conv2d_387616: 
quant_conv2d_387618: %
quant_dense_387624:	?
quant_dense_387626: 
quant_dense_387628:  
quant_dense_387630:
quant_dense_387632: 
quant_dense_387634: 
identity??$quant_conv2d/StatefulPartitionedCall?#quant_dense/StatefulPartitionedCall?%quant_dropout/StatefulPartitionedCall?&quantize_layer/StatefulPartitionedCall?
&quantize_layer/StatefulPartitionedCallStatefulPartitionedCallinputsquantize_layer_387603quantize_layer_387605*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(1* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_quantize_layer_layer_call_and_return_conditional_losses_3875582(
&quantize_layer/StatefulPartitionedCall?
$quant_conv2d/StatefulPartitionedCallStatefulPartitionedCall/quantize_layer/StatefulPartitionedCall:output:0quant_conv2d_387608quant_conv2d_387610quant_conv2d_387612quant_conv2d_387614quant_conv2d_387616quant_conv2d_387618*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_3875102&
$quant_conv2d/StatefulPartitionedCall?
#quant_max_pooling2d/PartitionedCallPartitionedCall-quant_conv2d/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_quant_max_pooling2d_layer_call_and_return_conditional_losses_3874382%
#quant_max_pooling2d/PartitionedCall?
%quant_dropout/StatefulPartitionedCallStatefulPartitionedCall,quant_max_pooling2d/PartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_quant_dropout_layer_call_and_return_conditional_losses_3874222'
%quant_dropout/StatefulPartitionedCall?
quant_flatten/PartitionedCallPartitionedCall.quant_dropout/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_quant_flatten_layer_call_and_return_conditional_losses_3873992
quant_flatten/PartitionedCall?
#quant_dense/StatefulPartitionedCallStatefulPartitionedCall&quant_flatten/PartitionedCall:output:0quant_dense_387624quant_dense_387626quant_dense_387628quant_dense_387630quant_dense_387632quant_dense_387634*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_quant_dense_layer_call_and_return_conditional_losses_3873702%
#quant_dense/StatefulPartitionedCall?
IdentityIdentity,quant_dense/StatefulPartitionedCall:output:0%^quant_conv2d/StatefulPartitionedCall$^quant_dense/StatefulPartitionedCall&^quant_dropout/StatefulPartitionedCall'^quantize_layer/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????(1: : : : : : : : : : : : : : 2L
$quant_conv2d/StatefulPartitionedCall$quant_conv2d/StatefulPartitionedCall2J
#quant_dense/StatefulPartitionedCall#quant_dense/StatefulPartitionedCall2N
%quant_dropout/StatefulPartitionedCall%quant_dropout/StatefulPartitionedCall2P
&quantize_layer/StatefulPartitionedCall&quantize_layer/StatefulPartitionedCall:W S
/
_output_shapes
:?????????(1
 
_user_specified_nameinputs
?
k
O__inference_quant_max_pooling2d_layer_call_and_return_conditional_losses_387211

inputs
identity?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????
*
ksize
*
paddingVALID*
strides
2	
MaxPooll
IdentityIdentityMaxPool:output:0*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
J
.__inference_quant_dropout_layer_call_fn_388237

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_quant_dropout_layer_call_and_return_conditional_losses_3872182
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?
e
I__inference_quant_flatten_layer_call_and_return_conditional_losses_388275

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_387143

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?L
?
__inference__traced_save_388525
file_prefix@
<savev2_quantize_layer_quantize_layer_min_read_readvariableop@
<savev2_quantize_layer_quantize_layer_max_read_readvariableop<
8savev2_quantize_layer_optimizer_step_read_readvariableop:
6savev2_quant_conv2d_optimizer_step_read_readvariableop6
2savev2_quant_conv2d_kernel_min_read_readvariableop6
2savev2_quant_conv2d_kernel_max_read_readvariableop?
;savev2_quant_conv2d_post_activation_min_read_readvariableop?
;savev2_quant_conv2d_post_activation_max_read_readvariableopA
=savev2_quant_max_pooling2d_optimizer_step_read_readvariableop;
7savev2_quant_dropout_optimizer_step_read_readvariableop;
7savev2_quant_flatten_optimizer_step_read_readvariableop9
5savev2_quant_dense_optimizer_step_read_readvariableop5
1savev2_quant_dense_kernel_min_read_readvariableop5
1savev2_quant_dense_kernel_max_read_readvariableop>
:savev2_quant_dense_post_activation_min_read_readvariableop>
:savev2_quant_dense_post_activation_max_read_readvariableop%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_conv2d_bias_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*?
value?B?&BBlayer_with_weights-0/quantize_layer_min/.ATTRIBUTES/VARIABLE_VALUEBBlayer_with_weights-0/quantize_layer_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-0/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-1/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-1/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-1/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-2/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-3/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-4/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB>layer_with_weights-5/optimizer_step/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/kernel_min/.ATTRIBUTES/VARIABLE_VALUEB:layer_with_weights-5/kernel_max/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-5/post_activation_min/.ATTRIBUTES/VARIABLE_VALUEBClayer_with_weights-5/post_activation_max/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB&variables/3/.ATTRIBUTES/VARIABLE_VALUEB&variables/4/.ATTRIBUTES/VARIABLE_VALUEB'variables/13/.ATTRIBUTES/VARIABLE_VALUEB'variables/14/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBBvariables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBBvariables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBCvariables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:&*
dtype0*_
valueVBT&B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0<savev2_quantize_layer_quantize_layer_min_read_readvariableop<savev2_quantize_layer_quantize_layer_max_read_readvariableop8savev2_quantize_layer_optimizer_step_read_readvariableop6savev2_quant_conv2d_optimizer_step_read_readvariableop2savev2_quant_conv2d_kernel_min_read_readvariableop2savev2_quant_conv2d_kernel_max_read_readvariableop;savev2_quant_conv2d_post_activation_min_read_readvariableop;savev2_quant_conv2d_post_activation_max_read_readvariableop=savev2_quant_max_pooling2d_optimizer_step_read_readvariableop7savev2_quant_dropout_optimizer_step_read_readvariableop7savev2_quant_flatten_optimizer_step_read_readvariableop5savev2_quant_dense_optimizer_step_read_readvariableop1savev2_quant_dense_kernel_min_read_readvariableop1savev2_quant_dense_kernel_max_read_readvariableop:savev2_quant_dense_post_activation_min_read_readvariableop:savev2_quant_dense_post_activation_max_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_conv2d_bias_read_readvariableop(savev2_conv2d_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *4
dtypes*
(2&	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : ::: : : : : : : : : : : : : : : ::
::	?: : : : ::
::	?::
::	?: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
:
: 

_output_shapes
::%!

_output_shapes
:	?:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: : 

_output_shapes
::,(
&
_output_shapes
:
:  

_output_shapes
::%!!

_output_shapes
:	?: "

_output_shapes
::,#(
&
_output_shapes
:
: $

_output_shapes
::%%!

_output_shapes
:	?:&

_output_shapes
: 
?
?
/__inference_quantize_layer_layer_call_fn_388078

inputs
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????(1* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *S
fNRL
J__inference_quantize_layer_layer_call_and_return_conditional_losses_3875582
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????(12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(1: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????(1
 
_user_specified_nameinputs
?'
?
J__inference_quantize_layer_layer_call_and_return_conditional_losses_388108

inputs;
1allvaluesquantize_minimum_readvariableop_resource: ;
1allvaluesquantize_maximum_readvariableop_resource: 
identity??#AllValuesQuantize/AssignMaxAllValue?#AllValuesQuantize/AssignMinAllValue?8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?(AllValuesQuantize/Maximum/ReadVariableOp?(AllValuesQuantize/Minimum/ReadVariableOp?
AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
AllValuesQuantize/Const?
AllValuesQuantize/BatchMinMininputs AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/BatchMin?
AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
AllValuesQuantize/Const_1?
AllValuesQuantize/BatchMaxMaxinputs"AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/BatchMax?
(AllValuesQuantize/Minimum/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype02*
(AllValuesQuantize/Minimum/ReadVariableOp?
AllValuesQuantize/MinimumMinimum0AllValuesQuantize/Minimum/ReadVariableOp:value:0#AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Minimum?
AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
AllValuesQuantize/Minimum_1/y?
AllValuesQuantize/Minimum_1MinimumAllValuesQuantize/Minimum:z:0&AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Minimum_1?
(AllValuesQuantize/Maximum/ReadVariableOpReadVariableOp1allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype02*
(AllValuesQuantize/Maximum/ReadVariableOp?
AllValuesQuantize/MaximumMaximum0AllValuesQuantize/Maximum/ReadVariableOp:value:0#AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Maximum?
AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
AllValuesQuantize/Maximum_1/y?
AllValuesQuantize/Maximum_1MaximumAllValuesQuantize/Maximum:z:0&AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Maximum_1?
#AllValuesQuantize/AssignMinAllValueAssignVariableOp1allvaluesquantize_minimum_readvariableop_resourceAllValuesQuantize/Minimum_1:z:0)^AllValuesQuantize/Minimum/ReadVariableOp*
_output_shapes
 *
dtype02%
#AllValuesQuantize/AssignMinAllValue?
#AllValuesQuantize/AssignMaxAllValueAssignVariableOp1allvaluesquantize_maximum_readvariableop_resourceAllValuesQuantize/Maximum_1:z:0)^AllValuesQuantize/Maximum/ReadVariableOp*
_output_shapes
 *
dtype02%
#AllValuesQuantize/AssignMaxAllValue?
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource$^AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype02:
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp1allvaluesquantize_maximum_readvariableop_resource$^AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype02<
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????(12+
)AllValuesQuantize/FakeQuantWithMinMaxVars?
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0$^AllValuesQuantize/AssignMaxAllValue$^AllValuesQuantize/AssignMinAllValue9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1)^AllValuesQuantize/Maximum/ReadVariableOp)^AllValuesQuantize/Minimum/ReadVariableOp*
T0*/
_output_shapes
:?????????(12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(1: : 2J
#AllValuesQuantize/AssignMaxAllValue#AllValuesQuantize/AssignMaxAllValue2J
#AllValuesQuantize/AssignMinAllValue#AllValuesQuantize/AssignMinAllValue2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12T
(AllValuesQuantize/Maximum/ReadVariableOp(AllValuesQuantize/Maximum/ReadVariableOp2T
(AllValuesQuantize/Minimum/ReadVariableOp(AllValuesQuantize/Minimum/ReadVariableOp:W S
/
_output_shapes
:?????????(1
 
_user_specified_nameinputs
?
e
I__inference_quant_flatten_layer_call_and_return_conditional_losses_388281

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:??????????2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
-__inference_quant_conv2d_layer_call_fn_388125

inputs!
unknown:

	unknown_0:
	unknown_1:
	unknown_2:
	unknown_3: 
	unknown_4: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Q
fLRJ
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_3871922
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????(1: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????(1
 
_user_specified_nameinputs
?'
?
J__inference_quantize_layer_layer_call_and_return_conditional_losses_387558

inputs;
1allvaluesquantize_minimum_readvariableop_resource: ;
1allvaluesquantize_maximum_readvariableop_resource: 
identity??#AllValuesQuantize/AssignMaxAllValue?#AllValuesQuantize/AssignMinAllValue?8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?(AllValuesQuantize/Maximum/ReadVariableOp?(AllValuesQuantize/Minimum/ReadVariableOp?
AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2
AllValuesQuantize/Const?
AllValuesQuantize/BatchMinMininputs AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/BatchMin?
AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2
AllValuesQuantize/Const_1?
AllValuesQuantize/BatchMaxMaxinputs"AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/BatchMax?
(AllValuesQuantize/Minimum/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype02*
(AllValuesQuantize/Minimum/ReadVariableOp?
AllValuesQuantize/MinimumMinimum0AllValuesQuantize/Minimum/ReadVariableOp:value:0#AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Minimum?
AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
AllValuesQuantize/Minimum_1/y?
AllValuesQuantize/Minimum_1MinimumAllValuesQuantize/Minimum:z:0&AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Minimum_1?
(AllValuesQuantize/Maximum/ReadVariableOpReadVariableOp1allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype02*
(AllValuesQuantize/Maximum/ReadVariableOp?
AllValuesQuantize/MaximumMaximum0AllValuesQuantize/Maximum/ReadVariableOp:value:0#AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Maximum?
AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
AllValuesQuantize/Maximum_1/y?
AllValuesQuantize/Maximum_1MaximumAllValuesQuantize/Maximum:z:0&AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: 2
AllValuesQuantize/Maximum_1?
#AllValuesQuantize/AssignMinAllValueAssignVariableOp1allvaluesquantize_minimum_readvariableop_resourceAllValuesQuantize/Minimum_1:z:0)^AllValuesQuantize/Minimum/ReadVariableOp*
_output_shapes
 *
dtype02%
#AllValuesQuantize/AssignMinAllValue?
#AllValuesQuantize/AssignMaxAllValueAssignVariableOp1allvaluesquantize_maximum_readvariableop_resourceAllValuesQuantize/Maximum_1:z:0)^AllValuesQuantize/Maximum/ReadVariableOp*
_output_shapes
 *
dtype02%
#AllValuesQuantize/AssignMaxAllValue?
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp1allvaluesquantize_minimum_readvariableop_resource$^AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype02:
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp1allvaluesquantize_maximum_readvariableop_resource$^AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype02<
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
)AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputs@AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BAllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????(12+
)AllValuesQuantize/FakeQuantWithMinMaxVars?
IdentityIdentity3AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0$^AllValuesQuantize/AssignMaxAllValue$^AllValuesQuantize/AssignMinAllValue9^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1)^AllValuesQuantize/Maximum/ReadVariableOp)^AllValuesQuantize/Minimum/ReadVariableOp*
T0*/
_output_shapes
:?????????(12

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????(1: : 2J
#AllValuesQuantize/AssignMaxAllValue#AllValuesQuantize/AssignMaxAllValue2J
#AllValuesQuantize/AssignMinAllValue#AllValuesQuantize/AssignMinAllValue2t
8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12T
(AllValuesQuantize/Maximum/ReadVariableOp(AllValuesQuantize/Maximum/ReadVariableOp2T
(AllValuesQuantize/Minimum/ReadVariableOp(AllValuesQuantize/Minimum/ReadVariableOp:W S
/
_output_shapes
:?????????(1
 
_user_specified_nameinputs
??
?
P__inference_ExtAudioDataModelQAT_layer_call_and_return_conditional_losses_388060

inputsJ
@quantize_layer_allvaluesquantize_minimum_readvariableop_resource: J
@quantize_layer_allvaluesquantize_maximum_readvariableop_resource: V
<quant_conv2d_lastvaluequant_batchmin_readvariableop_resource:
@
2quant_conv2d_lastvaluequant_assignminlast_resource:@
2quant_conv2d_lastvaluequant_assignmaxlast_resource::
,quant_conv2d_biasadd_readvariableop_resource:M
Cquant_conv2d_movingavgquantize_assignminema_readvariableop_resource: M
Cquant_conv2d_movingavgquantize_assignmaxema_readvariableop_resource: J
7quant_dense_lastvaluequant_rank_readvariableop_resource:	?;
1quant_dense_lastvaluequant_assignminlast_resource: ;
1quant_dense_lastvaluequant_assignmaxlast_resource: 9
+quant_dense_biasadd_readvariableop_resource:L
Bquant_dense_movingavgquantize_assignminema_readvariableop_resource: L
Bquant_dense_movingavgquantize_assignmaxema_readvariableop_resource: 
identity??#quant_conv2d/BiasAdd/ReadVariableOp?)quant_conv2d/LastValueQuant/AssignMaxLast?)quant_conv2d/LastValueQuant/AssignMinLast?3quant_conv2d/LastValueQuant/BatchMax/ReadVariableOp?3quant_conv2d/LastValueQuant/BatchMin/ReadVariableOp?Lquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1?Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2??quant_conv2d/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp?:quant_conv2d/MovingAvgQuantize/AssignMaxEma/ReadVariableOp??quant_conv2d/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp?:quant_conv2d/MovingAvgQuantize/AssignMinEma/ReadVariableOp?Equant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?"quant_dense/BiasAdd/ReadVariableOp?(quant_dense/LastValueQuant/AssignMaxLast?(quant_dense/LastValueQuant/AssignMinLast?2quant_dense/LastValueQuant/BatchMax/ReadVariableOp?2quant_dense/LastValueQuant/BatchMin/ReadVariableOp?Aquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp?Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1?Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2?>quant_dense/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp?9quant_dense/MovingAvgQuantize/AssignMaxEma/ReadVariableOp?>quant_dense/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp?9quant_dense/MovingAvgQuantize/AssignMinEma/ReadVariableOp?Dquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?Fquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?2quantize_layer/AllValuesQuantize/AssignMaxAllValue?2quantize_layer/AllValuesQuantize/AssignMinAllValue?Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?7quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp?7quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp?
&quantize_layer/AllValuesQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2(
&quantize_layer/AllValuesQuantize/Const?
)quantize_layer/AllValuesQuantize/BatchMinMininputs/quantize_layer/AllValuesQuantize/Const:output:0*
T0*
_output_shapes
: 2+
)quantize_layer/AllValuesQuantize/BatchMin?
(quantize_layer/AllValuesQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2*
(quantize_layer/AllValuesQuantize/Const_1?
)quantize_layer/AllValuesQuantize/BatchMaxMaxinputs1quantize_layer/AllValuesQuantize/Const_1:output:0*
T0*
_output_shapes
: 2+
)quantize_layer/AllValuesQuantize/BatchMax?
7quantize_layer/AllValuesQuantize/Minimum/ReadVariableOpReadVariableOp@quantize_layer_allvaluesquantize_minimum_readvariableop_resource*
_output_shapes
: *
dtype029
7quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp?
(quantize_layer/AllValuesQuantize/MinimumMinimum?quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp:value:02quantize_layer/AllValuesQuantize/BatchMin:output:0*
T0*
_output_shapes
: 2*
(quantize_layer/AllValuesQuantize/Minimum?
,quantize_layer/AllValuesQuantize/Minimum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,quantize_layer/AllValuesQuantize/Minimum_1/y?
*quantize_layer/AllValuesQuantize/Minimum_1Minimum,quantize_layer/AllValuesQuantize/Minimum:z:05quantize_layer/AllValuesQuantize/Minimum_1/y:output:0*
T0*
_output_shapes
: 2,
*quantize_layer/AllValuesQuantize/Minimum_1?
7quantize_layer/AllValuesQuantize/Maximum/ReadVariableOpReadVariableOp@quantize_layer_allvaluesquantize_maximum_readvariableop_resource*
_output_shapes
: *
dtype029
7quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp?
(quantize_layer/AllValuesQuantize/MaximumMaximum?quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp:value:02quantize_layer/AllValuesQuantize/BatchMax:output:0*
T0*
_output_shapes
: 2*
(quantize_layer/AllValuesQuantize/Maximum?
,quantize_layer/AllValuesQuantize/Maximum_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,quantize_layer/AllValuesQuantize/Maximum_1/y?
*quantize_layer/AllValuesQuantize/Maximum_1Maximum,quantize_layer/AllValuesQuantize/Maximum:z:05quantize_layer/AllValuesQuantize/Maximum_1/y:output:0*
T0*
_output_shapes
: 2,
*quantize_layer/AllValuesQuantize/Maximum_1?
2quantize_layer/AllValuesQuantize/AssignMinAllValueAssignVariableOp@quantize_layer_allvaluesquantize_minimum_readvariableop_resource.quantize_layer/AllValuesQuantize/Minimum_1:z:08^quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp*
_output_shapes
 *
dtype024
2quantize_layer/AllValuesQuantize/AssignMinAllValue?
2quantize_layer/AllValuesQuantize/AssignMaxAllValueAssignVariableOp@quantize_layer_allvaluesquantize_maximum_readvariableop_resource.quantize_layer/AllValuesQuantize/Maximum_1:z:08^quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp*
_output_shapes
 *
dtype024
2quantize_layer/AllValuesQuantize/AssignMaxAllValue?
Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp@quantize_layer_allvaluesquantize_minimum_readvariableop_resource3^quantize_layer/AllValuesQuantize/AssignMinAllValue*
_output_shapes
: *
dtype02I
Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@quantize_layer_allvaluesquantize_maximum_readvariableop_resource3^quantize_layer/AllValuesQuantize/AssignMaxAllValue*
_output_shapes
: *
dtype02K
Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
8quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsinputsOquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Qquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????(12:
8quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars?
3quant_conv2d/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp<quant_conv2d_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:
*
dtype025
3quant_conv2d/LastValueQuant/BatchMin/ReadVariableOp?
6quant_conv2d/LastValueQuant/BatchMin/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          28
6quant_conv2d/LastValueQuant/BatchMin/reduction_indices?
$quant_conv2d/LastValueQuant/BatchMinMin;quant_conv2d/LastValueQuant/BatchMin/ReadVariableOp:value:0?quant_conv2d/LastValueQuant/BatchMin/reduction_indices:output:0*
T0*
_output_shapes
:2&
$quant_conv2d/LastValueQuant/BatchMin?
3quant_conv2d/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp<quant_conv2d_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:
*
dtype025
3quant_conv2d/LastValueQuant/BatchMax/ReadVariableOp?
6quant_conv2d/LastValueQuant/BatchMax/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          28
6quant_conv2d/LastValueQuant/BatchMax/reduction_indices?
$quant_conv2d/LastValueQuant/BatchMaxMax;quant_conv2d/LastValueQuant/BatchMax/ReadVariableOp:value:0?quant_conv2d/LastValueQuant/BatchMax/reduction_indices:output:0*
T0*
_output_shapes
:2&
$quant_conv2d/LastValueQuant/BatchMax?
%quant_conv2d/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%quant_conv2d/LastValueQuant/truediv/y?
#quant_conv2d/LastValueQuant/truedivRealDiv-quant_conv2d/LastValueQuant/BatchMax:output:0.quant_conv2d/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
:2%
#quant_conv2d/LastValueQuant/truediv?
#quant_conv2d/LastValueQuant/MinimumMinimum-quant_conv2d/LastValueQuant/BatchMin:output:0'quant_conv2d/LastValueQuant/truediv:z:0*
T0*
_output_shapes
:2%
#quant_conv2d/LastValueQuant/Minimum?
!quant_conv2d/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2#
!quant_conv2d/LastValueQuant/mul/y?
quant_conv2d/LastValueQuant/mulMul-quant_conv2d/LastValueQuant/BatchMin:output:0*quant_conv2d/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
:2!
quant_conv2d/LastValueQuant/mul?
#quant_conv2d/LastValueQuant/MaximumMaximum-quant_conv2d/LastValueQuant/BatchMax:output:0#quant_conv2d/LastValueQuant/mul:z:0*
T0*
_output_shapes
:2%
#quant_conv2d/LastValueQuant/Maximum?
)quant_conv2d/LastValueQuant/AssignMinLastAssignVariableOp2quant_conv2d_lastvaluequant_assignminlast_resource'quant_conv2d/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02+
)quant_conv2d/LastValueQuant/AssignMinLast?
)quant_conv2d/LastValueQuant/AssignMaxLastAssignVariableOp2quant_conv2d_lastvaluequant_assignmaxlast_resource'quant_conv2d/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02+
)quant_conv2d/LastValueQuant/AssignMaxLast?
Lquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOp<quant_conv2d_lastvaluequant_batchmin_readvariableop_resource*&
_output_shapes
:
*
dtype02N
Lquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOp2quant_conv2d_lastvaluequant_assignminlast_resource*^quant_conv2d/LastValueQuant/AssignMinLast*
_output_shapes
:*
dtype02P
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1?
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOp2quant_conv2d_lastvaluequant_assignmaxlast_resource*^quant_conv2d/LastValueQuant/AssignMaxLast*
_output_shapes
:*
dtype02P
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2?
=quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelTquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0Vquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0Vquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:
*
narrow_range(2?
=quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel?
quant_conv2d/Conv2DConv2DBquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars:outputs:0Gquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
quant_conv2d/Conv2D?
#quant_conv2d/BiasAdd/ReadVariableOpReadVariableOp,quant_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02%
#quant_conv2d/BiasAdd/ReadVariableOp?
quant_conv2d/BiasAddBiasAddquant_conv2d/Conv2D:output:0+quant_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
quant_conv2d/BiasAdd?
quant_conv2d/ReluReluquant_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
quant_conv2d/Relu?
$quant_conv2d/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*%
valueB"             2&
$quant_conv2d/MovingAvgQuantize/Const?
'quant_conv2d/MovingAvgQuantize/BatchMinMinquant_conv2d/Relu:activations:0-quant_conv2d/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2)
'quant_conv2d/MovingAvgQuantize/BatchMin?
&quant_conv2d/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*%
valueB"             2(
&quant_conv2d/MovingAvgQuantize/Const_1?
'quant_conv2d/MovingAvgQuantize/BatchMaxMaxquant_conv2d/Relu:activations:0/quant_conv2d/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2)
'quant_conv2d/MovingAvgQuantize/BatchMax?
(quant_conv2d/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(quant_conv2d/MovingAvgQuantize/Minimum/y?
&quant_conv2d/MovingAvgQuantize/MinimumMinimum0quant_conv2d/MovingAvgQuantize/BatchMin:output:01quant_conv2d/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2(
&quant_conv2d/MovingAvgQuantize/Minimum?
(quant_conv2d/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2*
(quant_conv2d/MovingAvgQuantize/Maximum/y?
&quant_conv2d/MovingAvgQuantize/MaximumMaximum0quant_conv2d/MovingAvgQuantize/BatchMax:output:01quant_conv2d/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2(
&quant_conv2d/MovingAvgQuantize/Maximum?
1quant_conv2d/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o?:23
1quant_conv2d/MovingAvgQuantize/AssignMinEma/decay?
:quant_conv2d/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpCquant_conv2d_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02<
:quant_conv2d/MovingAvgQuantize/AssignMinEma/ReadVariableOp?
/quant_conv2d/MovingAvgQuantize/AssignMinEma/subSubBquant_conv2d/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0*quant_conv2d/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 21
/quant_conv2d/MovingAvgQuantize/AssignMinEma/sub?
/quant_conv2d/MovingAvgQuantize/AssignMinEma/mulMul3quant_conv2d/MovingAvgQuantize/AssignMinEma/sub:z:0:quant_conv2d/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 21
/quant_conv2d/MovingAvgQuantize/AssignMinEma/mul?
?quant_conv2d/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpCquant_conv2d_movingavgquantize_assignminema_readvariableop_resource3quant_conv2d/MovingAvgQuantize/AssignMinEma/mul:z:0;^quant_conv2d/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype02A
?quant_conv2d/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp?
1quant_conv2d/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o?:23
1quant_conv2d/MovingAvgQuantize/AssignMaxEma/decay?
:quant_conv2d/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpCquant_conv2d_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02<
:quant_conv2d/MovingAvgQuantize/AssignMaxEma/ReadVariableOp?
/quant_conv2d/MovingAvgQuantize/AssignMaxEma/subSubBquant_conv2d/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0*quant_conv2d/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 21
/quant_conv2d/MovingAvgQuantize/AssignMaxEma/sub?
/quant_conv2d/MovingAvgQuantize/AssignMaxEma/mulMul3quant_conv2d/MovingAvgQuantize/AssignMaxEma/sub:z:0:quant_conv2d/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 21
/quant_conv2d/MovingAvgQuantize/AssignMaxEma/mul?
?quant_conv2d/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpCquant_conv2d_movingavgquantize_assignmaxema_readvariableop_resource3quant_conv2d/MovingAvgQuantize/AssignMaxEma/mul:z:0;^quant_conv2d/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype02A
?quant_conv2d/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp?
Equant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpCquant_conv2d_movingavgquantize_assignminema_readvariableop_resource@^quant_conv2d/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02G
Equant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCquant_conv2d_movingavgquantize_assignmaxema_readvariableop_resource@^quant_conv2d/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02I
Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
6quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_conv2d/Relu:activations:0Mquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Oquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????28
6quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars?
quant_max_pooling2d/MaxPoolMaxPool@quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0*/
_output_shapes
:?????????
*
ksize
*
paddingVALID*
strides
2
quant_max_pooling2d/MaxPool
quant_dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
quant_dropout/dropout/Const?
quant_dropout/dropout/MulMul$quant_max_pooling2d/MaxPool:output:0$quant_dropout/dropout/Const:output:0*
T0*/
_output_shapes
:?????????
2
quant_dropout/dropout/Mul?
quant_dropout/dropout/ShapeShape$quant_max_pooling2d/MaxPool:output:0*
T0*
_output_shapes
:2
quant_dropout/dropout/Shape?
2quant_dropout/dropout/random_uniform/RandomUniformRandomUniform$quant_dropout/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????
*
dtype0*
seed??24
2quant_dropout/dropout/random_uniform/RandomUniform?
$quant_dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2&
$quant_dropout/dropout/GreaterEqual/y?
"quant_dropout/dropout/GreaterEqualGreaterEqual;quant_dropout/dropout/random_uniform/RandomUniform:output:0-quant_dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????
2$
"quant_dropout/dropout/GreaterEqual?
quant_dropout/dropout/CastCast&quant_dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????
2
quant_dropout/dropout/Cast?
quant_dropout/dropout/Mul_1Mulquant_dropout/dropout/Mul:z:0quant_dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????
2
quant_dropout/dropout/Mul_1{
quant_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2
quant_flatten/Const?
quant_flatten/ReshapeReshapequant_dropout/dropout/Mul_1:z:0quant_flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
quant_flatten/Reshape?
.quant_dense/LastValueQuant/Rank/ReadVariableOpReadVariableOp7quant_dense_lastvaluequant_rank_readvariableop_resource*
_output_shapes
:	?*
dtype020
.quant_dense/LastValueQuant/Rank/ReadVariableOp?
quant_dense/LastValueQuant/RankConst*
_output_shapes
: *
dtype0*
value	B :2!
quant_dense/LastValueQuant/Rank?
&quant_dense/LastValueQuant/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2(
&quant_dense/LastValueQuant/range/start?
&quant_dense/LastValueQuant/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2(
&quant_dense/LastValueQuant/range/delta?
 quant_dense/LastValueQuant/rangeRange/quant_dense/LastValueQuant/range/start:output:0(quant_dense/LastValueQuant/Rank:output:0/quant_dense/LastValueQuant/range/delta:output:0*
_output_shapes
:2"
 quant_dense/LastValueQuant/range?
2quant_dense/LastValueQuant/BatchMin/ReadVariableOpReadVariableOp7quant_dense_lastvaluequant_rank_readvariableop_resource*
_output_shapes
:	?*
dtype024
2quant_dense/LastValueQuant/BatchMin/ReadVariableOp?
#quant_dense/LastValueQuant/BatchMinMin:quant_dense/LastValueQuant/BatchMin/ReadVariableOp:value:0)quant_dense/LastValueQuant/range:output:0*
T0*
_output_shapes
: 2%
#quant_dense/LastValueQuant/BatchMin?
0quant_dense/LastValueQuant/Rank_1/ReadVariableOpReadVariableOp7quant_dense_lastvaluequant_rank_readvariableop_resource*
_output_shapes
:	?*
dtype022
0quant_dense/LastValueQuant/Rank_1/ReadVariableOp?
!quant_dense/LastValueQuant/Rank_1Const*
_output_shapes
: *
dtype0*
value	B :2#
!quant_dense/LastValueQuant/Rank_1?
(quant_dense/LastValueQuant/range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2*
(quant_dense/LastValueQuant/range_1/start?
(quant_dense/LastValueQuant/range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2*
(quant_dense/LastValueQuant/range_1/delta?
"quant_dense/LastValueQuant/range_1Range1quant_dense/LastValueQuant/range_1/start:output:0*quant_dense/LastValueQuant/Rank_1:output:01quant_dense/LastValueQuant/range_1/delta:output:0*
_output_shapes
:2$
"quant_dense/LastValueQuant/range_1?
2quant_dense/LastValueQuant/BatchMax/ReadVariableOpReadVariableOp7quant_dense_lastvaluequant_rank_readvariableop_resource*
_output_shapes
:	?*
dtype024
2quant_dense/LastValueQuant/BatchMax/ReadVariableOp?
#quant_dense/LastValueQuant/BatchMaxMax:quant_dense/LastValueQuant/BatchMax/ReadVariableOp:value:0+quant_dense/LastValueQuant/range_1:output:0*
T0*
_output_shapes
: 2%
#quant_dense/LastValueQuant/BatchMax?
$quant_dense/LastValueQuant/truediv/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2&
$quant_dense/LastValueQuant/truediv/y?
"quant_dense/LastValueQuant/truedivRealDiv,quant_dense/LastValueQuant/BatchMax:output:0-quant_dense/LastValueQuant/truediv/y:output:0*
T0*
_output_shapes
: 2$
"quant_dense/LastValueQuant/truediv?
"quant_dense/LastValueQuant/MinimumMinimum,quant_dense/LastValueQuant/BatchMin:output:0&quant_dense/LastValueQuant/truediv:z:0*
T0*
_output_shapes
: 2$
"quant_dense/LastValueQuant/Minimum?
 quant_dense/LastValueQuant/mul/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2"
 quant_dense/LastValueQuant/mul/y?
quant_dense/LastValueQuant/mulMul,quant_dense/LastValueQuant/BatchMin:output:0)quant_dense/LastValueQuant/mul/y:output:0*
T0*
_output_shapes
: 2 
quant_dense/LastValueQuant/mul?
"quant_dense/LastValueQuant/MaximumMaximum,quant_dense/LastValueQuant/BatchMax:output:0"quant_dense/LastValueQuant/mul:z:0*
T0*
_output_shapes
: 2$
"quant_dense/LastValueQuant/Maximum?
(quant_dense/LastValueQuant/AssignMinLastAssignVariableOp1quant_dense_lastvaluequant_assignminlast_resource&quant_dense/LastValueQuant/Minimum:z:0*
_output_shapes
 *
dtype02*
(quant_dense/LastValueQuant/AssignMinLast?
(quant_dense/LastValueQuant/AssignMaxLastAssignVariableOp1quant_dense_lastvaluequant_assignmaxlast_resource&quant_dense/LastValueQuant/Maximum:z:0*
_output_shapes
 *
dtype02*
(quant_dense/LastValueQuant/AssignMaxLast?
Aquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp7quant_dense_lastvaluequant_rank_readvariableop_resource*
_output_shapes
:	?*
dtype02C
Aquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp?
Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp1quant_dense_lastvaluequant_assignminlast_resource)^quant_dense/LastValueQuant/AssignMinLast*
_output_shapes
: *
dtype02E
Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1?
Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp1quant_dense_lastvaluequant_assignmaxlast_resource)^quant_dense/LastValueQuant/AssignMaxLast*
_output_shapes
: *
dtype02E
Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2?
2quant_dense/LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsIquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Kquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0Kquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	?*
narrow_range(24
2quant_dense/LastValueQuant/FakeQuantWithMinMaxVars?
quant_dense/MatMulMatMulquant_flatten/Reshape:output:0<quant_dense/LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:?????????2
quant_dense/MatMul?
"quant_dense/BiasAdd/ReadVariableOpReadVariableOp+quant_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02$
"quant_dense/BiasAdd/ReadVariableOp?
quant_dense/BiasAddBiasAddquant_dense/MatMul:product:0*quant_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
quant_dense/BiasAdd?
#quant_dense/MovingAvgQuantize/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2%
#quant_dense/MovingAvgQuantize/Const?
&quant_dense/MovingAvgQuantize/BatchMinMinquant_dense/BiasAdd:output:0,quant_dense/MovingAvgQuantize/Const:output:0*
T0*
_output_shapes
: 2(
&quant_dense/MovingAvgQuantize/BatchMin?
%quant_dense/MovingAvgQuantize/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2'
%quant_dense/MovingAvgQuantize/Const_1?
&quant_dense/MovingAvgQuantize/BatchMaxMaxquant_dense/BiasAdd:output:0.quant_dense/MovingAvgQuantize/Const_1:output:0*
T0*
_output_shapes
: 2(
&quant_dense/MovingAvgQuantize/BatchMax?
'quant_dense/MovingAvgQuantize/Minimum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'quant_dense/MovingAvgQuantize/Minimum/y?
%quant_dense/MovingAvgQuantize/MinimumMinimum/quant_dense/MovingAvgQuantize/BatchMin:output:00quant_dense/MovingAvgQuantize/Minimum/y:output:0*
T0*
_output_shapes
: 2'
%quant_dense/MovingAvgQuantize/Minimum?
'quant_dense/MovingAvgQuantize/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2)
'quant_dense/MovingAvgQuantize/Maximum/y?
%quant_dense/MovingAvgQuantize/MaximumMaximum/quant_dense/MovingAvgQuantize/BatchMax:output:00quant_dense/MovingAvgQuantize/Maximum/y:output:0*
T0*
_output_shapes
: 2'
%quant_dense/MovingAvgQuantize/Maximum?
0quant_dense/MovingAvgQuantize/AssignMinEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0quant_dense/MovingAvgQuantize/AssignMinEma/decay?
9quant_dense/MovingAvgQuantize/AssignMinEma/ReadVariableOpReadVariableOpBquant_dense_movingavgquantize_assignminema_readvariableop_resource*
_output_shapes
: *
dtype02;
9quant_dense/MovingAvgQuantize/AssignMinEma/ReadVariableOp?
.quant_dense/MovingAvgQuantize/AssignMinEma/subSubAquant_dense/MovingAvgQuantize/AssignMinEma/ReadVariableOp:value:0)quant_dense/MovingAvgQuantize/Minimum:z:0*
T0*
_output_shapes
: 20
.quant_dense/MovingAvgQuantize/AssignMinEma/sub?
.quant_dense/MovingAvgQuantize/AssignMinEma/mulMul2quant_dense/MovingAvgQuantize/AssignMinEma/sub:z:09quant_dense/MovingAvgQuantize/AssignMinEma/decay:output:0*
T0*
_output_shapes
: 20
.quant_dense/MovingAvgQuantize/AssignMinEma/mul?
>quant_dense/MovingAvgQuantize/AssignMinEma/AssignSubVariableOpAssignSubVariableOpBquant_dense_movingavgquantize_assignminema_readvariableop_resource2quant_dense/MovingAvgQuantize/AssignMinEma/mul:z:0:^quant_dense/MovingAvgQuantize/AssignMinEma/ReadVariableOp*
_output_shapes
 *
dtype02@
>quant_dense/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp?
0quant_dense/MovingAvgQuantize/AssignMaxEma/decayConst*
_output_shapes
: *
dtype0*
valueB
 *o?:22
0quant_dense/MovingAvgQuantize/AssignMaxEma/decay?
9quant_dense/MovingAvgQuantize/AssignMaxEma/ReadVariableOpReadVariableOpBquant_dense_movingavgquantize_assignmaxema_readvariableop_resource*
_output_shapes
: *
dtype02;
9quant_dense/MovingAvgQuantize/AssignMaxEma/ReadVariableOp?
.quant_dense/MovingAvgQuantize/AssignMaxEma/subSubAquant_dense/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:value:0)quant_dense/MovingAvgQuantize/Maximum:z:0*
T0*
_output_shapes
: 20
.quant_dense/MovingAvgQuantize/AssignMaxEma/sub?
.quant_dense/MovingAvgQuantize/AssignMaxEma/mulMul2quant_dense/MovingAvgQuantize/AssignMaxEma/sub:z:09quant_dense/MovingAvgQuantize/AssignMaxEma/decay:output:0*
T0*
_output_shapes
: 20
.quant_dense/MovingAvgQuantize/AssignMaxEma/mul?
>quant_dense/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOpAssignSubVariableOpBquant_dense_movingavgquantize_assignmaxema_readvariableop_resource2quant_dense/MovingAvgQuantize/AssignMaxEma/mul:z:0:^quant_dense/MovingAvgQuantize/AssignMaxEma/ReadVariableOp*
_output_shapes
 *
dtype02@
>quant_dense/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp?
Dquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpBquant_dense_movingavgquantize_assignminema_readvariableop_resource?^quant_dense/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp*
_output_shapes
: *
dtype02F
Dquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
Fquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpBquant_dense_movingavgquantize_assignmaxema_readvariableop_resource?^quant_dense/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp*
_output_shapes
: *
dtype02H
Fquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
5quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsquant_dense/BiasAdd:output:0Lquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0Nquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:?????????27
5quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars?
IdentityIdentity?quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0$^quant_conv2d/BiasAdd/ReadVariableOp*^quant_conv2d/LastValueQuant/AssignMaxLast*^quant_conv2d/LastValueQuant/AssignMinLast4^quant_conv2d/LastValueQuant/BatchMax/ReadVariableOp4^quant_conv2d/LastValueQuant/BatchMin/ReadVariableOpM^quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpO^quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1O^quant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2@^quant_conv2d/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp;^quant_conv2d/MovingAvgQuantize/AssignMaxEma/ReadVariableOp@^quant_conv2d/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp;^quant_conv2d/MovingAvgQuantize/AssignMinEma/ReadVariableOpF^quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpH^quant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1#^quant_dense/BiasAdd/ReadVariableOp)^quant_dense/LastValueQuant/AssignMaxLast)^quant_dense/LastValueQuant/AssignMinLast3^quant_dense/LastValueQuant/BatchMax/ReadVariableOp3^quant_dense/LastValueQuant/BatchMin/ReadVariableOpB^quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpD^quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1D^quant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2?^quant_dense/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp:^quant_dense/MovingAvgQuantize/AssignMaxEma/ReadVariableOp?^quant_dense/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp:^quant_dense/MovingAvgQuantize/AssignMinEma/ReadVariableOpE^quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpG^quant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_13^quantize_layer/AllValuesQuantize/AssignMaxAllValue3^quantize_layer/AllValuesQuantize/AssignMinAllValueH^quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpJ^quantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_18^quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp8^quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????(1: : : : : : : : : : : : : : 2J
#quant_conv2d/BiasAdd/ReadVariableOp#quant_conv2d/BiasAdd/ReadVariableOp2V
)quant_conv2d/LastValueQuant/AssignMaxLast)quant_conv2d/LastValueQuant/AssignMaxLast2V
)quant_conv2d/LastValueQuant/AssignMinLast)quant_conv2d/LastValueQuant/AssignMinLast2j
3quant_conv2d/LastValueQuant/BatchMax/ReadVariableOp3quant_conv2d/LastValueQuant/BatchMax/ReadVariableOp2j
3quant_conv2d/LastValueQuant/BatchMin/ReadVariableOp3quant_conv2d/LastValueQuant/BatchMin/ReadVariableOp2?
Lquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpLquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2?
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12?
Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2Nquant_conv2d/LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22?
?quant_conv2d/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp?quant_conv2d/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2x
:quant_conv2d/MovingAvgQuantize/AssignMaxEma/ReadVariableOp:quant_conv2d/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2?
?quant_conv2d/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp?quant_conv2d/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2x
:quant_conv2d/MovingAvgQuantize/AssignMinEma/ReadVariableOp:quant_conv2d/MovingAvgQuantize/AssignMinEma/ReadVariableOp2?
Equant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpEquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2?
Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Gquant_conv2d/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12H
"quant_dense/BiasAdd/ReadVariableOp"quant_dense/BiasAdd/ReadVariableOp2T
(quant_dense/LastValueQuant/AssignMaxLast(quant_dense/LastValueQuant/AssignMaxLast2T
(quant_dense/LastValueQuant/AssignMinLast(quant_dense/LastValueQuant/AssignMinLast2h
2quant_dense/LastValueQuant/BatchMax/ReadVariableOp2quant_dense/LastValueQuant/BatchMax/ReadVariableOp2h
2quant_dense/LastValueQuant/BatchMin/ReadVariableOp2quant_dense/LastValueQuant/BatchMin/ReadVariableOp2?
Aquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpAquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2?
Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12?
Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2Cquant_dense/LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22?
>quant_dense/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp>quant_dense/MovingAvgQuantize/AssignMaxEma/AssignSubVariableOp2v
9quant_dense/MovingAvgQuantize/AssignMaxEma/ReadVariableOp9quant_dense/MovingAvgQuantize/AssignMaxEma/ReadVariableOp2?
>quant_dense/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp>quant_dense/MovingAvgQuantize/AssignMinEma/AssignSubVariableOp2v
9quant_dense/MovingAvgQuantize/AssignMinEma/ReadVariableOp9quant_dense/MovingAvgQuantize/AssignMinEma/ReadVariableOp2?
Dquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpDquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2?
Fquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Fquant_dense/MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12h
2quantize_layer/AllValuesQuantize/AssignMaxAllValue2quantize_layer/AllValuesQuantize/AssignMaxAllValue2h
2quantize_layer/AllValuesQuantize/AssignMinAllValue2quantize_layer/AllValuesQuantize/AssignMinAllValue2?
Gquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOpGquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2?
Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1Iquantize_layer/AllValuesQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp7quantize_layer/AllValuesQuantize/Maximum/ReadVariableOp2r
7quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp7quantize_layer/AllValuesQuantize/Minimum/ReadVariableOp:W S
/
_output_shapes
:?????????(1
 
_user_specified_nameinputs
?
?
5__inference_ExtAudioDataModelQAT_layer_call_fn_387852

inputs
unknown: 
	unknown_0: #
	unknown_1:

	unknown_2:
	unknown_3:
	unknown_4:
	unknown_5: 
	unknown_6: 
	unknown_7:	?
	unknown_8: 
	unknown_9: 

unknown_10:

unknown_11: 

unknown_12: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*0
_read_only_resource_inputs
	
*-
config_proto

CPU

GPU 2J 8? *Y
fTRR
P__inference_ExtAudioDataModelQAT_layer_call_and_return_conditional_losses_3872632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????(1: : : : : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????(1
 
_user_specified_nameinputs
?&
?
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_388163

inputsb
Hlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource:
X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource:X
Jlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource:-
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity??BiasAdd/ReadVariableOp??LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1?ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2?8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpReadVariableOpHlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_resource*&
_output_shapes
:
*
dtype02A
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_1_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ReadVariableOpJlastvaluequant_fakequantwithminmaxvarsperchannel_readvariableop_2_resource*
_output_shapes
:*
dtype02C
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2?
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel!FakeQuantWithMinMaxVarsPerChannelGLastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1:value:0ILastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2:value:0*&
_output_shapes
:
*
narrow_range(22
0LastValueQuant/FakeQuantWithMinMaxVarsPerChannel?
Conv2DConv2Dinputs:LastValueQuant/FakeQuantWithMinMaxVarsPerChannel:outputs:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu?
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsRelu:activations:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*/
_output_shapes
:?????????2+
)MovingAvgQuantize/FakeQuantWithMinMaxVars?
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp@^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOpB^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1B^LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*:
_input_shapes)
':?????????(1: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2?
?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp?LastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp2?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_1ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_12?
ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_2ALastValueQuant/FakeQuantWithMinMaxVarsPerChannel/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:W S
/
_output_shapes
:?????????(1
 
_user_specified_nameinputs
?"
?
G__inference_quant_dense_layer_call_and_return_conditional_losses_388335

inputsQ
>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource:	?J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource: J
@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource: -
biasadd_readvariableop_resource:K
Amovingavgquantize_fakequantwithminmaxvars_readvariableop_resource: M
Cmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource: 
identity??BiasAdd/ReadVariableOp?5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp?7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1?7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2?8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOp>lastvaluequant_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
:	?*
dtype027
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp?
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype029
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1?
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2ReadVariableOp@lastvaluequant_fakequantwithminmaxvars_readvariableop_2_resource*
_output_shapes
: *
dtype029
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2?
&LastValueQuant/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVars=LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0?LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_2:value:0*
_output_shapes
:	?*
narrow_range(2(
&LastValueQuant/FakeQuantWithMinMaxVars?
MatMulMatMulinputs0LastValueQuant/FakeQuantWithMinMaxVars:outputs:0*
T0*'
_output_shapes
:?????????2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2	
BiasAdd?
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOpReadVariableOpAmovingavgquantize_fakequantwithminmaxvars_readvariableop_resource*
_output_shapes
: *
dtype02:
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp?
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1ReadVariableOpCmovingavgquantize_fakequantwithminmaxvars_readvariableop_1_resource*
_output_shapes
: *
dtype02<
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1?
)MovingAvgQuantize/FakeQuantWithMinMaxVarsFakeQuantWithMinMaxVarsBiasAdd:output:0@MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp:value:0BMovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:value:0*'
_output_shapes
:?????????2+
)MovingAvgQuantize/FakeQuantWithMinMaxVars?
IdentityIdentity3MovingAvgQuantize/FakeQuantWithMinMaxVars:outputs:0^BiasAdd/ReadVariableOp6^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp8^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_18^LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_29^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp;^MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*3
_input_shapes"
 :??????????: : : : : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2n
5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp5LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp2r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_17LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_12r
7LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_27LastValueQuant/FakeQuantWithMinMaxVars/ReadVariableOp_22t
8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp8MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp2x
:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:MovingAvgQuantize/FakeQuantWithMinMaxVars/ReadVariableOp_1:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
h
I__inference_quant_dropout_layer_call_and_return_conditional_losses_387422

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????
2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????
*
dtype0*
seed??2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????
2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????
2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????
2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?
P
4__inference_quant_max_pooling2d_layer_call_fn_388217

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *X
fSRQ
O__inference_quant_max_pooling2d_layer_call_and_return_conditional_losses_3872112
PartitionedCallt
IdentityIdentityPartitionedCall:output:0*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
g
I__inference_quant_dropout_layer_call_and_return_conditional_losses_387218

inputs

identity_1b
IdentityIdentityinputs*
T0*/
_output_shapes
:?????????
2

Identityq

Identity_1IdentityIdentity:output:0*
T0*/
_output_shapes
:?????????
2

Identity_1"!

identity_1Identity_1:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?
h
I__inference_quant_dropout_layer_call_and_return_conditional_losses_388259

inputs
identity?c
dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/Const{
dropout/MulMulinputsdropout/Const:output:0*
T0*/
_output_shapes
:?????????
2
dropout/MulT
dropout/ShapeShapeinputs*
T0*
_output_shapes
:2
dropout/Shape?
$dropout/random_uniform/RandomUniformRandomUniformdropout/Shape:output:0*
T0*/
_output_shapes
:?????????
*
dtype0*
seed??2&
$dropout/random_uniform/RandomUniformu
dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2
dropout/GreaterEqual/y?
dropout/GreaterEqualGreaterEqual-dropout/random_uniform/RandomUniform:output:0dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????
2
dropout/GreaterEqual?
dropout/CastCastdropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????
2
dropout/Cast?
dropout/Mul_1Muldropout/Mul:z:0dropout/Cast:y:0*
T0*/
_output_shapes
:?????????
2
dropout/Mul_1m
IdentityIdentitydropout/Mul_1:z:0*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????
:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?
J
.__inference_max_pooling2d_layer_call_fn_387149

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *R
fMRK
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_3871432
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????(1?
quant_dense0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?I
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
layer_with_weights-4
layer-4
layer_with_weights-5
layer-5
	optimizer
	variables
	trainable_variables

regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"?F
_tf_keras_sequential?F{"name": "ExtAudioDataModelQAT", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "ExtAudioDataModelQAT", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 49, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}, {"class_name": "QuantizeLayer", "config": {"name": "quantize_layer", "trainable": true, "dtype": "float32", "quantizer": {"class_name": "AllValuesQuantizer", "config": {"num_bits": 8, "per_axis": false, "symmetric": false, "narrow_range": false}}}}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_conv2d", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [8, 10]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "relu"}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_max_pooling2d", "trainable": true, "dtype": "float32", "layer": {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, "quantize_config": {"class_name": "Default8BitQuantizeConfig", "config": {"weight_attrs": [], "activation_attrs": [], "quantize_output": false}}}}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_dropout", "trainable": true, "dtype": "float32", "layer": {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, "quantize_config": {"class_name": "Default8BitQuantizeConfig", "config": {"weight_attrs": [], "activation_attrs": [], "quantize_output": false}}}}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_flatten", "trainable": true, "dtype": "float32", "layer": {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, "quantize_config": {"class_name": "Default8BitQuantizeConfig", "config": {"weight_attrs": [], "activation_attrs": [], "quantize_output": false}}}}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_dense", "trainable": true, "dtype": "float32", "layer": {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 4, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "linear"}}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "quantize_config": {"class_name": "Default8BitQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}}]}, "shared_object_id": 24, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40, 49, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 40, 49, 1]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "ExtAudioDataModelQAT", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 49, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "shared_object_id": 0}, {"class_name": "QuantizeLayer", "config": {"name": "quantize_layer", "trainable": true, "dtype": "float32", "quantizer": {"class_name": "AllValuesQuantizer", "config": {"num_bits": 8, "per_axis": false, "symmetric": false, "narrow_range": false}}}, "shared_object_id": 2}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_conv2d", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [8, 10]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "relu"}, "shared_object_id": 3}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}, "shared_object_id": 8}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_max_pooling2d", "trainable": true, "dtype": "float32", "layer": {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 9}, "quantize_config": {"class_name": "Default8BitQuantizeConfig", "config": {"weight_attrs": [], "activation_attrs": [], "quantize_output": false}}}, "shared_object_id": 11}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_dropout", "trainable": true, "dtype": "float32", "layer": {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "shared_object_id": 12}, "quantize_config": {"class_name": "Default8BitQuantizeConfig", "config": {"weight_attrs": [], "activation_attrs": [], "quantize_output": false}}}, "shared_object_id": 14}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_flatten", "trainable": true, "dtype": "float32", "layer": {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 15}, "quantize_config": {"class_name": "Default8BitQuantizeConfig", "config": {"weight_attrs": [], "activation_attrs": [], "quantize_output": false}}}, "shared_object_id": 17}, {"class_name": "QuantizeWrapper", "config": {"name": "quant_dense", "trainable": true, "dtype": "float32", "layer": {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 4, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "linear"}, "shared_object_id": 18}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 21}, "quantize_config": {"class_name": "Default8BitQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}, "shared_object_id": 23}]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": true}, "shared_object_id": 31}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 32}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0009897876298055053, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
quantize_layer_min
quantize_layer_max
quantizer_vars
optimizer_step
	variables
trainable_variables
regularization_losses
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "quantize_layer", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "QuantizeLayer", "config": {"name": "quantize_layer", "trainable": true, "dtype": "float32", "quantizer": {"class_name": "AllValuesQuantizer", "config": {"num_bits": 8, "per_axis": false, "symmetric": false, "narrow_range": false}}}, "shared_object_id": 2, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40, 49, 1]}}
?
	layer
optimizer_step
_weight_vars

kernel_min

kernel_max
_quantize_activations
post_activation_min
post_activation_max
_output_quantizers
	variables
trainable_variables
 regularization_losses
!	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?
{"name": "quant_conv2d", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "QuantizeWrapper", "config": {"name": "quant_conv2d", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [8, 10]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "relu"}, "shared_object_id": 3}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6}, "quantize_config": {"class_name": "Default8BitConvQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}, "shared_object_id": 8, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40, 49, 1]}}
?
	"layer
#optimizer_step
$_weight_vars
%_quantize_activations
&_output_quantizers
'	variables
(trainable_variables
)regularization_losses
*	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "quant_max_pooling2d", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "QuantizeWrapper", "config": {"name": "quant_max_pooling2d", "trainable": true, "dtype": "float32", "layer": {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 9}, "quantize_config": {"class_name": "Default8BitQuantizeConfig", "config": {"weight_attrs": [], "activation_attrs": [], "quantize_output": false}}}, "shared_object_id": 11, "build_input_shape": {"class_name": "TensorShape", "items": [null, 17, 20, 8]}}
?
	+layer
,optimizer_step
-_weight_vars
._quantize_activations
/_output_quantizers
0	variables
1trainable_variables
2regularization_losses
3	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "quant_dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "QuantizeWrapper", "config": {"name": "quant_dropout", "trainable": true, "dtype": "float32", "layer": {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "shared_object_id": 12}, "quantize_config": {"class_name": "Default8BitQuantizeConfig", "config": {"weight_attrs": [], "activation_attrs": [], "quantize_output": false}}}, "shared_object_id": 14, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 10, 8]}}
?
	4layer
5optimizer_step
6_weight_vars
7_quantize_activations
8_output_quantizers
9	variables
:trainable_variables
;regularization_losses
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "quant_flatten", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "QuantizeWrapper", "config": {"name": "quant_flatten", "trainable": true, "dtype": "float32", "layer": {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 15}, "quantize_config": {"class_name": "Default8BitQuantizeConfig", "config": {"weight_attrs": [], "activation_attrs": [], "quantize_output": false}}}, "shared_object_id": 17, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 10, 8]}}
?
	=layer
>optimizer_step
?_weight_vars
@
kernel_min
A
kernel_max
B_quantize_activations
Cpost_activation_min
Dpost_activation_max
E_output_quantizers
F	variables
Gtrainable_variables
Hregularization_losses
I	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"name": "quant_dense", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "QuantizeWrapper", "config": {"name": "quant_dense", "trainable": true, "dtype": "float32", "layer": {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 4, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "linear"}, "shared_object_id": 18}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 21}, "quantize_config": {"class_name": "Default8BitQuantizeConfig", "config": {"weight_attrs": ["kernel"], "activation_attrs": ["activation"], "quantize_output": false}}}, "shared_object_id": 23, "build_input_shape": {"class_name": "TensorShape", "items": [null, 640]}}
?

Jbeta_1

Kbeta_2
	Ldecay
Mlearning_rate
NiterOm?Pm?Qm?Rm?Ov?Pv?Qv?Rv?"
	optimizer
?
0
1
2
O3
P4
5
6
7
8
9
#10
,11
512
Q13
R14
>15
@16
A17
C18
D19"
trackable_list_wrapper
<
O0
P1
Q2
R3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Slayer_metrics
Tmetrics
	variables

Ulayers
Vlayer_regularization_losses
	trainable_variables

regularization_losses
Wnon_trainable_variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
):' 2!quantize_layer/quantize_layer_min
):' 2!quantize_layer/quantize_layer_max
:
min_var
max_var"
trackable_dict_wrapper
%:# 2quantize_layer/optimizer_step
5
0
1
2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Xlayer_metrics
Ymetrics
	variables

Zlayers
[layer_regularization_losses
trainable_variables
regularization_losses
\non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

Pkernel
Obias
]	variables
^trainable_variables
_regularization_losses
`	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?

_tf_keras_layer?	{"name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [8, 10]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "relu"}, "shared_object_id": 3}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 4}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 5}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 39}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40, 49, 1]}}
#:! 2quant_conv2d/optimizer_step
'
a0"
trackable_list_wrapper
#:!2quant_conv2d/kernel_min
#:!2quant_conv2d/kernel_max
 "
trackable_list_wrapper
(:& 2 quant_conv2d/post_activation_min
(:& 2 quant_conv2d/post_activation_max
 "
trackable_list_wrapper
Q
O0
P1
2
3
4
5
6"
trackable_list_wrapper
.
O0
P1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
blayer_metrics
cmetrics
	variables

dlayers
elayer_regularization_losses
trainable_variables
 regularization_losses
fnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
g	variables
htrainable_variables
iregularization_losses
j	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 40}}
*:( 2"quant_max_pooling2d/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
#0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
klayer_metrics
lmetrics
'	variables

mlayers
nlayer_regularization_losses
(trainable_variables
)regularization_losses
onon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
p	variables
qtrainable_variables
rregularization_losses
s	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "shared_object_id": 12}
$:" 2quant_dropout/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
,0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
tlayer_metrics
umetrics
0	variables

vlayers
wlayer_regularization_losses
1trainable_variables
2regularization_losses
xnon_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
y	variables
ztrainable_variables
{regularization_losses
|	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 41}}
$:" 2quant_flatten/optimizer_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
50"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
}layer_metrics
~metrics
9	variables

layers
 ?layer_regularization_losses
:trainable_variables
;regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?	

Rkernel
Qbias
?	variables
?trainable_variables
?regularization_losses
?	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 4, "activation": {"class_name": "QuantizeAwareActivation", "config": {"activation": "linear"}, "shared_object_id": 18}, "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 19}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 20}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 21, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 640}}, "shared_object_id": 42}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 640]}}
":  2quant_dense/optimizer_step
(
?0"
trackable_list_wrapper
: 2quant_dense/kernel_min
: 2quant_dense/kernel_max
 "
trackable_list_wrapper
':% 2quant_dense/post_activation_min
':% 2quant_dense/post_activation_max
 "
trackable_list_wrapper
Q
Q0
R1
>2
@3
A4
C5
D6"
trackable_list_wrapper
.
Q0
R1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?metrics
F	variables
?layers
 ?layer_regularization_losses
Gtrainable_variables
Hregularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
:	 (2	Adam/iter
:2conv2d/bias
':%
2conv2d/kernel
:2
dense/bias
:	?2dense/kernel
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
3
4
5
6
7
#8
,9
510
>11
@12
A13
C14
D15"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
'
O0"
trackable_list_wrapper
'
O0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?metrics
]	variables
?layers
 ?layer_regularization_losses
^trainable_variables
_regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0
P0
?2"
trackable_tuple_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?metrics
g	variables
?layers
 ?layer_regularization_losses
htrainable_variables
iregularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
"0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
#0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?metrics
p	variables
?layers
 ?layer_regularization_losses
qtrainable_variables
rregularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
+0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
,0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?metrics
y	variables
?layers
 ?layer_regularization_losses
ztrainable_variables
{regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
40"
trackable_list_wrapper
 "
trackable_list_wrapper
'
50"
trackable_list_wrapper
'
Q0"
trackable_list_wrapper
'
Q0"
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layer_metrics
?metrics
?	variables
?layers
 ?layer_regularization_losses
?trainable_variables
?regularization_losses
?non_trainable_variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
0
R0
?2"
trackable_tuple_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
=0"
trackable_list_wrapper
 "
trackable_list_wrapper
C
>0
@1
A2
C3
D4"
trackable_list_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 43}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 32}
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
min_var
max_var"
trackable_dict_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:
@min_var
Amax_var"
trackable_dict_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
:2Adam/conv2d/bias/m
,:*
2Adam/conv2d/kernel/m
:2Adam/dense/bias/m
$:"	?2Adam/dense/kernel/m
:2Adam/conv2d/bias/v
,:*
2Adam/conv2d/kernel/v
:2Adam/dense/bias/v
$:"	?2Adam/dense/kernel/v
?2?
5__inference_ExtAudioDataModelQAT_layer_call_fn_387294
5__inference_ExtAudioDataModelQAT_layer_call_fn_387852
5__inference_ExtAudioDataModelQAT_layer_call_fn_387885
5__inference_ExtAudioDataModelQAT_layer_call_fn_387702?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
P__inference_ExtAudioDataModelQAT_layer_call_and_return_conditional_losses_387931
P__inference_ExtAudioDataModelQAT_layer_call_and_return_conditional_losses_388060
P__inference_ExtAudioDataModelQAT_layer_call_and_return_conditional_losses_387740
P__inference_ExtAudioDataModelQAT_layer_call_and_return_conditional_losses_387778?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
!__inference__wrapped_model_387137?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_1?????????(1
?2?
/__inference_quantize_layer_layer_call_fn_388069
/__inference_quantize_layer_layer_call_fn_388078?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
J__inference_quantize_layer_layer_call_and_return_conditional_losses_388087
J__inference_quantize_layer_layer_call_and_return_conditional_losses_388108?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
-__inference_quant_conv2d_layer_call_fn_388125
-__inference_quant_conv2d_layer_call_fn_388142?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_388163
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_388212?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
4__inference_quant_max_pooling2d_layer_call_fn_388217
4__inference_quant_max_pooling2d_layer_call_fn_388222?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
O__inference_quant_max_pooling2d_layer_call_and_return_conditional_losses_388227
O__inference_quant_max_pooling2d_layer_call_and_return_conditional_losses_388232?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_quant_dropout_layer_call_fn_388237
.__inference_quant_dropout_layer_call_fn_388242?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_quant_dropout_layer_call_and_return_conditional_losses_388247
I__inference_quant_dropout_layer_call_and_return_conditional_losses_388259?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
.__inference_quant_flatten_layer_call_fn_388264
.__inference_quant_flatten_layer_call_fn_388269?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
I__inference_quant_flatten_layer_call_and_return_conditional_losses_388275
I__inference_quant_flatten_layer_call_and_return_conditional_losses_388281?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
,__inference_quant_dense_layer_call_fn_388298
,__inference_quant_dense_layer_call_fn_388315?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
G__inference_quant_dense_layer_call_and_return_conditional_losses_388335
G__inference_quant_dense_layer_call_and_return_conditional_losses_388391?
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?B?
$__inference_signature_wrapper_387819input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
.__inference_max_pooling2d_layer_call_fn_387149?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_387143?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec)
args!?
jself
jinputs

jtraining
varargs
 
varkw
 
defaults?
p 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
P__inference_ExtAudioDataModelQAT_layer_call_and_return_conditional_losses_387740yPOR@AQCD@?=
6?3
)?&
input_1?????????(1
p 

 
? "%?"
?
0?????????
? ?
P__inference_ExtAudioDataModelQAT_layer_call_and_return_conditional_losses_387778yPOR@AQCD@?=
6?3
)?&
input_1?????????(1
p

 
? "%?"
?
0?????????
? ?
P__inference_ExtAudioDataModelQAT_layer_call_and_return_conditional_losses_387931xPOR@AQCD??<
5?2
(?%
inputs?????????(1
p 

 
? "%?"
?
0?????????
? ?
P__inference_ExtAudioDataModelQAT_layer_call_and_return_conditional_losses_388060xPOR@AQCD??<
5?2
(?%
inputs?????????(1
p

 
? "%?"
?
0?????????
? ?
5__inference_ExtAudioDataModelQAT_layer_call_fn_387294lPOR@AQCD@?=
6?3
)?&
input_1?????????(1
p 

 
? "???????????
5__inference_ExtAudioDataModelQAT_layer_call_fn_387702lPOR@AQCD@?=
6?3
)?&
input_1?????????(1
p

 
? "???????????
5__inference_ExtAudioDataModelQAT_layer_call_fn_387852kPOR@AQCD??<
5?2
(?%
inputs?????????(1
p 

 
? "???????????
5__inference_ExtAudioDataModelQAT_layer_call_fn_387885kPOR@AQCD??<
5?2
(?%
inputs?????????(1
p

 
? "???????????
!__inference__wrapped_model_387137?POR@AQCD8?5
.?+
)?&
input_1?????????(1
? "9?6
4
quant_dense%?"
quant_dense??????????
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_387143?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
.__inference_max_pooling2d_layer_call_fn_387149?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_388163tPO;?8
1?.
(?%
inputs?????????(1
p 
? "-?*
#? 
0?????????
? ?
H__inference_quant_conv2d_layer_call_and_return_conditional_losses_388212tPO;?8
1?.
(?%
inputs?????????(1
p
? "-?*
#? 
0?????????
? ?
-__inference_quant_conv2d_layer_call_fn_388125gPO;?8
1?.
(?%
inputs?????????(1
p 
? " ???????????
-__inference_quant_conv2d_layer_call_fn_388142gPO;?8
1?.
(?%
inputs?????????(1
p
? " ???????????
G__inference_quant_dense_layer_call_and_return_conditional_losses_388335eR@AQCD4?1
*?'
!?
inputs??????????
p 
? "%?"
?
0?????????
? ?
G__inference_quant_dense_layer_call_and_return_conditional_losses_388391eR@AQCD4?1
*?'
!?
inputs??????????
p
? "%?"
?
0?????????
? ?
,__inference_quant_dense_layer_call_fn_388298XR@AQCD4?1
*?'
!?
inputs??????????
p 
? "???????????
,__inference_quant_dense_layer_call_fn_388315XR@AQCD4?1
*?'
!?
inputs??????????
p
? "???????????
I__inference_quant_dropout_layer_call_and_return_conditional_losses_388247l;?8
1?.
(?%
inputs?????????

p 
? "-?*
#? 
0?????????

? ?
I__inference_quant_dropout_layer_call_and_return_conditional_losses_388259l;?8
1?.
(?%
inputs?????????

p
? "-?*
#? 
0?????????

? ?
.__inference_quant_dropout_layer_call_fn_388237_;?8
1?.
(?%
inputs?????????

p 
? " ??????????
?
.__inference_quant_dropout_layer_call_fn_388242_;?8
1?.
(?%
inputs?????????

p
? " ??????????
?
I__inference_quant_flatten_layer_call_and_return_conditional_losses_388275e;?8
1?.
(?%
inputs?????????

p 
? "&?#
?
0??????????
? ?
I__inference_quant_flatten_layer_call_and_return_conditional_losses_388281e;?8
1?.
(?%
inputs?????????

p
? "&?#
?
0??????????
? ?
.__inference_quant_flatten_layer_call_fn_388264X;?8
1?.
(?%
inputs?????????

p 
? "????????????
.__inference_quant_flatten_layer_call_fn_388269X;?8
1?.
(?%
inputs?????????

p
? "????????????
O__inference_quant_max_pooling2d_layer_call_and_return_conditional_losses_388227l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????

? ?
O__inference_quant_max_pooling2d_layer_call_and_return_conditional_losses_388232l;?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????

? ?
4__inference_quant_max_pooling2d_layer_call_fn_388217_;?8
1?.
(?%
inputs?????????
p 
? " ??????????
?
4__inference_quant_max_pooling2d_layer_call_fn_388222_;?8
1?.
(?%
inputs?????????
p
? " ??????????
?
J__inference_quantize_layer_layer_call_and_return_conditional_losses_388087p;?8
1?.
(?%
inputs?????????(1
p 
? "-?*
#? 
0?????????(1
? ?
J__inference_quantize_layer_layer_call_and_return_conditional_losses_388108p;?8
1?.
(?%
inputs?????????(1
p
? "-?*
#? 
0?????????(1
? ?
/__inference_quantize_layer_layer_call_fn_388069c;?8
1?.
(?%
inputs?????????(1
p 
? " ??????????(1?
/__inference_quantize_layer_layer_call_fn_388078c;?8
1?.
(?%
inputs?????????(1
p
? " ??????????(1?
$__inference_signature_wrapper_387819?POR@AQCDC?@
? 
9?6
4
input_1)?&
input_1?????????(1"9?6
4
quant_dense%?"
quant_dense?????????