??
??
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
>
Maximum
x"T
y"T
z"T"
Ttype:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
?
Mul
x"T
y"T
z"T"
Ttype:
2	?
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
@
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu6
features"T
activations"T"
Ttype:
2	
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
-
Sqrt
x"T
y"T"
Ttype:

2
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
<
Sub
x"T
y"T
z"T"
Ttype:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
`
meanVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namemean
Y
mean/Read/ReadVariableOpReadVariableOpmean*
_output_shapes
:*
dtype0
h
varianceVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
variance
a
variance/Read/ReadVariableOpReadVariableOpvariance*
_output_shapes
:*
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0	
?
prune_low_magnitude_conv2d/maskVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*0
shared_name!prune_low_magnitude_conv2d/mask
?
3prune_low_magnitude_conv2d/mask/Read/ReadVariableOpReadVariableOpprune_low_magnitude_conv2d/mask*&
_output_shapes
:
*
dtype0
?
$prune_low_magnitude_conv2d/thresholdVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$prune_low_magnitude_conv2d/threshold
?
8prune_low_magnitude_conv2d/threshold/Read/ReadVariableOpReadVariableOp$prune_low_magnitude_conv2d/threshold*
_output_shapes
: *
dtype0
?
'prune_low_magnitude_conv2d/pruning_stepVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *8
shared_name)'prune_low_magnitude_conv2d/pruning_step
?
;prune_low_magnitude_conv2d/pruning_step/Read/ReadVariableOpReadVariableOp'prune_low_magnitude_conv2d/pruning_step*
_output_shapes
: *
dtype0	
?
.prune_low_magnitude_max_pooling2d/pruning_stepVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *?
shared_name0.prune_low_magnitude_max_pooling2d/pruning_step
?
Bprune_low_magnitude_max_pooling2d/pruning_step/Read/ReadVariableOpReadVariableOp.prune_low_magnitude_max_pooling2d/pruning_step*
_output_shapes
: *
dtype0	
?
(prune_low_magnitude_flatten/pruning_stepVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *9
shared_name*(prune_low_magnitude_flatten/pruning_step
?
<prune_low_magnitude_flatten/pruning_step/Read/ReadVariableOpReadVariableOp(prune_low_magnitude_flatten/pruning_step*
_output_shapes
: *
dtype0	
?
prune_low_magnitude_dense/maskVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*/
shared_name prune_low_magnitude_dense/mask
?
2prune_low_magnitude_dense/mask/Read/ReadVariableOpReadVariableOpprune_low_magnitude_dense/mask*
_output_shapes
:	?*
dtype0
?
#prune_low_magnitude_dense/thresholdVarHandleOp*
_output_shapes
: *
dtype0*
shape: *4
shared_name%#prune_low_magnitude_dense/threshold
?
7prune_low_magnitude_dense/threshold/Read/ReadVariableOpReadVariableOp#prune_low_magnitude_dense/threshold*
_output_shapes
: *
dtype0
?
&prune_low_magnitude_dense/pruning_stepVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *7
shared_name(&prune_low_magnitude_dense/pruning_step
?
:prune_low_magnitude_dense/pruning_step/Read/ReadVariableOpReadVariableOp&prune_low_magnitude_dense/pruning_step*
_output_shapes
: *
dtype0	
\
iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_nameiter
U
iter/Read/ReadVariableOpReadVariableOpiter*
_output_shapes
: *
dtype0	
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
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
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

NoOpNoOp
?<
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?<
value?<B?< B?;
?
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
	optimizer
trainable_variables
	regularization_losses

	variables
	keras_api

signatures
?

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean
variance
	count
	keras_api
?
pruning_vars
	layer
prunable_weights
mask
	threshold
pruning_step
trainable_variables
regularization_losses
	variables
	keras_api
?
pruning_vars
	 layer
!prunable_weights
"pruning_step
#trainable_variables
$regularization_losses
%	variables
&	keras_api
R
'trainable_variables
(regularization_losses
)	variables
*	keras_api
?
+pruning_vars
	,layer
-prunable_weights
.pruning_step
/trainable_variables
0regularization_losses
1	variables
2	keras_api
?
3pruning_vars
	4layer
5prunable_weights
6mask
7	threshold
8pruning_step
9trainable_variables
:regularization_losses
;	variables
<	keras_api
?
=iter

>beta_1

?beta_2
	@decay
Alearning_rateBm?Cm?Dm?Em?Bv?Cv?Dv?Ev?

B0
C1
D2
E3
 
n
0
1
2
B3
C4
5
6
7
"8
.9
D10
E11
612
713
814
?
Fmetrics

Glayers
trainable_variables
Hnon_trainable_variables
Ilayer_regularization_losses
	regularization_losses
Jlayer_metrics

	variables
 
 
 
 
 
NL
VARIABLE_VALUEmean4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEvariance8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUEcount5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUE
 

K0
h

Bkernel
Cbias
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api

B0
ig
VARIABLE_VALUEprune_low_magnitude_conv2d/mask4layer_with_weights-1/mask/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUE$prune_low_magnitude_conv2d/threshold9layer_with_weights-1/threshold/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE'prune_low_magnitude_conv2d/pruning_step<layer_with_weights-1/pruning_step/.ATTRIBUTES/VARIABLE_VALUE

B0
C1
 
#
B0
C1
2
3
4
?
Pmetrics

Qlayers
trainable_variables
Rlayer_regularization_losses
Snon_trainable_variables
regularization_losses
Tlayer_metrics
	variables
 
R
Utrainable_variables
Vregularization_losses
W	variables
X	keras_api
 
?~
VARIABLE_VALUE.prune_low_magnitude_max_pooling2d/pruning_step<layer_with_weights-2/pruning_step/.ATTRIBUTES/VARIABLE_VALUE
 
 

"0
?
Ymetrics

Zlayers
#trainable_variables
[layer_regularization_losses
\non_trainable_variables
$regularization_losses
]layer_metrics
%	variables
 
 
 
?
^metrics

_layers
'trainable_variables
`layer_regularization_losses
anon_trainable_variables
(regularization_losses
blayer_metrics
)	variables
 
R
ctrainable_variables
dregularization_losses
e	variables
f	keras_api
 
zx
VARIABLE_VALUE(prune_low_magnitude_flatten/pruning_step<layer_with_weights-3/pruning_step/.ATTRIBUTES/VARIABLE_VALUE
 
 

.0
?
gmetrics

hlayers
/trainable_variables
ilayer_regularization_losses
jnon_trainable_variables
0regularization_losses
klayer_metrics
1	variables

l0
h

Dkernel
Ebias
mtrainable_variables
nregularization_losses
o	variables
p	keras_api

D0
hf
VARIABLE_VALUEprune_low_magnitude_dense/mask4layer_with_weights-4/mask/.ATTRIBUTES/VARIABLE_VALUE
rp
VARIABLE_VALUE#prune_low_magnitude_dense/threshold9layer_with_weights-4/threshold/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUE&prune_low_magnitude_dense/pruning_step<layer_with_weights-4/pruning_step/.ATTRIBUTES/VARIABLE_VALUE

D0
E1
 
#
D0
E1
62
73
84
?
qmetrics

rlayers
9trainable_variables
slayer_regularization_losses
tnon_trainable_variables
:regularization_losses
ulayer_metrics
;	variables
CA
VARIABLE_VALUEiter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
GE
VARIABLE_VALUEbeta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
EC
VARIABLE_VALUEdecay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUElearning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
SQ
VARIABLE_VALUEconv2d/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEconv2d/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
RP
VARIABLE_VALUEdense/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
PN
VARIABLE_VALUE
dense/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE

v0
w1
*
0
1
2
3
4
5
N
0
1
2
3
4
5
"6
.7
68
79
810
 
 

B0
1
2

B0
C1
 

B0
C1
?
xmetrics

ylayers
Ltrainable_variables
zlayer_regularization_losses
{non_trainable_variables
Mregularization_losses
|layer_metrics
N	variables
 

0
 

0
1
2
 
 
 
 
?
}metrics

~layers
Utrainable_variables
layer_regularization_losses
?non_trainable_variables
Vregularization_losses
?layer_metrics
W	variables
 

 0
 

"0
 
 
 
 
 
 
 
 
 
?
?metrics
?layers
ctrainable_variables
 ?layer_regularization_losses
?non_trainable_variables
dregularization_losses
?layer_metrics
e	variables
 

,0
 

.0
 

D0
61
72

D0
E1
 

D0
E1
?
?metrics
?layers
mtrainable_variables
 ?layer_regularization_losses
?non_trainable_variables
nregularization_losses
?layer_metrics
o	variables
 

40
 

60
71
82
 
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
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

?0
?1

?	variables
vt
VARIABLE_VALUEAdam/conv2d/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/conv2d/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/dense/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
vt
VARIABLE_VALUEAdam/conv2d/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
tr
VARIABLE_VALUEAdam/conv2d/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUEAdam/dense/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
sq
VARIABLE_VALUEAdam/dense/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????(1*
dtype0*$
shape:?????????(1
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1meanvarianceconv2d/kernelprune_low_magnitude_conv2d/maskconv2d/biasdense/kernelprune_low_magnitude_dense/mask
dense/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *-
f(R&
$__inference_signature_wrapper_246955
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenamemean/Read/ReadVariableOpvariance/Read/ReadVariableOpcount/Read/ReadVariableOp3prune_low_magnitude_conv2d/mask/Read/ReadVariableOp8prune_low_magnitude_conv2d/threshold/Read/ReadVariableOp;prune_low_magnitude_conv2d/pruning_step/Read/ReadVariableOpBprune_low_magnitude_max_pooling2d/pruning_step/Read/ReadVariableOp<prune_low_magnitude_flatten/pruning_step/Read/ReadVariableOp2prune_low_magnitude_dense/mask/Read/ReadVariableOp7prune_low_magnitude_dense/threshold/Read/ReadVariableOp:prune_low_magnitude_dense/pruning_step/Read/ReadVariableOpiter/Read/ReadVariableOpbeta_1/Read/ReadVariableOpbeta_2/Read/ReadVariableOpdecay/Read/ReadVariableOp!learning_rate/Read/ReadVariableOp!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_2/Read/ReadVariableOp(Adam/conv2d/kernel/m/Read/ReadVariableOp&Adam/conv2d/bias/m/Read/ReadVariableOp'Adam/dense/kernel/m/Read/ReadVariableOp%Adam/dense/bias/m/Read/ReadVariableOp(Adam/conv2d/kernel/v/Read/ReadVariableOp&Adam/conv2d/bias/v/Read/ReadVariableOp'Adam/dense/kernel/v/Read/ReadVariableOp%Adam/dense/bias/v/Read/ReadVariableOpConst*-
Tin&
$2"						*
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
__inference__traced_save_248164
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamemeanvariancecountprune_low_magnitude_conv2d/mask$prune_low_magnitude_conv2d/threshold'prune_low_magnitude_conv2d/pruning_step.prune_low_magnitude_max_pooling2d/pruning_step(prune_low_magnitude_flatten/pruning_stepprune_low_magnitude_dense/mask#prune_low_magnitude_dense/threshold&prune_low_magnitude_dense/pruning_stepiterbeta_1beta_2decaylearning_rateconv2d/kernelconv2d/biasdense/kernel
dense/biastotalcount_1total_1count_2Adam/conv2d/kernel/mAdam/conv2d/bias/mAdam/dense/kernel/mAdam/dense/bias/mAdam/conv2d/kernel/vAdam/conv2d/bias/vAdam/dense/kernel/vAdam/dense/bias/v*,
Tin%
#2!*
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
"__inference__traced_restore_248270??
?
?
,prune_low_magnitude_conv2d_cond_false_247121/
+prune_low_magnitude_conv2d_cond_placeholder1
-prune_low_magnitude_conv2d_cond_placeholder_11
-prune_low_magnitude_conv2d_cond_placeholder_21
-prune_low_magnitude_conv2d_cond_placeholder_3T
Pprune_low_magnitude_conv2d_cond_identity_prune_low_magnitude_conv2d_logicaland_1
.
*prune_low_magnitude_conv2d_cond_identity_1
j
$prune_low_magnitude_conv2d/cond/NoOpNoOp*
_output_shapes
 2&
$prune_low_magnitude_conv2d/cond/NoOp?
(prune_low_magnitude_conv2d/cond/IdentityIdentityPprune_low_magnitude_conv2d_cond_identity_prune_low_magnitude_conv2d_logicaland_1%^prune_low_magnitude_conv2d/cond/NoOp*
T0
*
_output_shapes
: 2*
(prune_low_magnitude_conv2d/cond/Identity?
*prune_low_magnitude_conv2d/cond/Identity_1Identity1prune_low_magnitude_conv2d/cond/Identity:output:0*
T0
*
_output_shapes
: 2,
*prune_low_magnitude_conv2d/cond/Identity_1"a
*prune_low_magnitude_conv2d_cond_identity_13prune_low_magnitude_conv2d/cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : :

_output_shapes
: 
?
D
(__inference_dropout_layer_call_fn_247766

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
GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_2461602
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
?
?
;__inference_prune_low_magnitude_conv2d_layer_call_fn_247515

inputs!
unknown:
#
	unknown_0:

	unknown_1:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_prune_low_magnitude_conv2d_layer_call_and_return_conditional_losses_2461402
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????(1: : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????(1
 
_user_specified_nameinputs
?
?
4assert_greater_equal_Assert_AssertGuard_false_246392K
Gassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all
V
Rassert_greater_equal_assert_assertguard_assert_assert_greater_equal_readvariableop	I
Eassert_greater_equal_assert_assertguard_assert_assert_greater_equal_y	6
2assert_greater_equal_assert_assertguard_identity_1
??.assert_greater_equal/Assert/AssertGuard/Assert?
5assert_greater_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*?
value?B? B?Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.27
5assert_greater_equal/Assert/AssertGuard/Assert/data_0?
5assert_greater_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:27
5assert_greater_equal/Assert/AssertGuard/Assert/data_1?
5assert_greater_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 27
5assert_greater_equal/Assert/AssertGuard/Assert/data_2?
5assert_greater_equal/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 27
5assert_greater_equal/Assert/AssertGuard/Assert/data_4?
.assert_greater_equal/Assert/AssertGuard/AssertAssertGassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all>assert_greater_equal/Assert/AssertGuard/Assert/data_0:output:0>assert_greater_equal/Assert/AssertGuard/Assert/data_1:output:0>assert_greater_equal/Assert/AssertGuard/Assert/data_2:output:0Rassert_greater_equal_assert_assertguard_assert_assert_greater_equal_readvariableop>assert_greater_equal/Assert/AssertGuard/Assert/data_4:output:0Eassert_greater_equal_assert_assertguard_assert_assert_greater_equal_y*
T

2		*
_output_shapes
 20
.assert_greater_equal/Assert/AssertGuard/Assert?
0assert_greater_equal/Assert/AssertGuard/IdentityIdentityGassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all/^assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identity?
2assert_greater_equal/Assert/AssertGuard/Identity_1Identity9assert_greater_equal/Assert/AssertGuard/Identity:output:0/^assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 24
2assert_greater_equal/Assert/AssertGuard/Identity_1"q
2assert_greater_equal_assert_assertguard_identity_1;assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2`
.assert_greater_equal/Assert/AssertGuard/Assert.assert_greater_equal/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
2prune_low_magnitude_max_pooling2d_cond_true_247245b
^prune_low_magnitude_max_pooling2d_cond_identity_prune_low_magnitude_max_pooling2d_logicaland_1
5
1prune_low_magnitude_max_pooling2d_cond_identity_1
?
1prune_low_magnitude_max_pooling2d/cond/group_depsNoOp*
_output_shapes
 23
1prune_low_magnitude_max_pooling2d/cond/group_deps?
/prune_low_magnitude_max_pooling2d/cond/IdentityIdentity^prune_low_magnitude_max_pooling2d_cond_identity_prune_low_magnitude_max_pooling2d_logicaland_12^prune_low_magnitude_max_pooling2d/cond/group_deps*
T0
*
_output_shapes
: 21
/prune_low_magnitude_max_pooling2d/cond/Identity?
1prune_low_magnitude_max_pooling2d/cond/Identity_1Identity8prune_low_magnitude_max_pooling2d/cond/Identity:output:0*
T0
*
_output_shapes
: 23
1prune_low_magnitude_max_pooling2d/cond/Identity_1"o
1prune_low_magnitude_max_pooling2d_cond_identity_1:prune_low_magnitude_max_pooling2d/cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
e
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_246099

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
?7
?
cond_true_2476013
)cond_greaterequal_readvariableop_resource:	 F
,cond_pruning_ops_abs_readvariableop_resource:
8
cond_assignvariableop_resource:
*
 cond_assignvariableop_1_resource: 
cond_identity_logicaland_1

cond_identity_1
??cond/AssignVariableOp?cond/AssignVariableOp_1? cond/GreaterEqual/ReadVariableOp?cond/LessEqual/ReadVariableOp?cond/Sub/ReadVariableOp?#cond/pruning_ops/Abs/ReadVariableOp?
 cond/GreaterEqual/ReadVariableOpReadVariableOp)cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	2"
 cond/GreaterEqual/ReadVariableOpl
cond/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
cond/GreaterEqual/y?
cond/GreaterEqualGreaterEqual(cond/GreaterEqual/ReadVariableOp:value:0cond/GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
cond/GreaterEqual?
cond/LessEqual/ReadVariableOpReadVariableOp)cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	2
cond/LessEqual/ReadVariableOpo
cond/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2
cond/LessEqual/y?
cond/LessEqual	LessEqual%cond/LessEqual/ReadVariableOp:value:0cond/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
cond/LessEquale
cond/Less/xConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
cond/Less/x\
cond/Less/yConst*
_output_shapes
: *
dtype0*
value	B : 2
cond/Less/yk
	cond/LessLesscond/Less/x:output:0cond/Less/y:output:0*
T0*
_output_shapes
: 2
	cond/Lessh
cond/LogicalOr	LogicalOrcond/LessEqual:z:0cond/Less:z:0*
_output_shapes
: 2
cond/LogicalOrs
cond/LogicalAnd
LogicalAndcond/GreaterEqual:z:0cond/LogicalOr:z:0*
_output_shapes
: 2
cond/LogicalAnd?
cond/Sub/ReadVariableOpReadVariableOp)cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	2
cond/Sub/ReadVariableOpZ

cond/Sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2

cond/Sub/yr
cond/SubSubcond/Sub/ReadVariableOp:value:0cond/Sub/y:output:0*
T0	*
_output_shapes
: 2

cond/Subd
cond/FloorMod/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rd2
cond/FloorMod/ys
cond/FloorModFloorModcond/Sub:z:0cond/FloorMod/y:output:0*
T0	*
_output_shapes
: 2
cond/FloorMod^
cond/Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
cond/Equal/yl

cond/EqualEqualcond/FloorMod:z:0cond/Equal/y:output:0*
T0	*
_output_shapes
: 2

cond/Equalq
cond/LogicalAnd_1
LogicalAndcond/LogicalAnd:z:0cond/Equal:z:0*
_output_shapes
: 2
cond/LogicalAnd_1]

cond/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

cond/Const?
#cond/pruning_ops/Abs/ReadVariableOpReadVariableOp,cond_pruning_ops_abs_readvariableop_resource*&
_output_shapes
:
*
dtype02%
#cond/pruning_ops/Abs/ReadVariableOp?
cond/pruning_ops/AbsAbs+cond/pruning_ops/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:
2
cond/pruning_ops/Absq
cond/pruning_ops/SizeConst*
_output_shapes
: *
dtype0*
value
B :?2
cond/pruning_ops/Size?
cond/pruning_ops/CastCastcond/pruning_ops/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
cond/pruning_ops/Castu
cond/pruning_ops/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
cond/pruning_ops/sub/x?
cond/pruning_ops/subSubcond/pruning_ops/sub/x:output:0cond/Const:output:0*
T0*
_output_shapes
: 2
cond/pruning_ops/sub?
cond/pruning_ops/mulMulcond/pruning_ops/Cast:y:0cond/pruning_ops/sub:z:0*
T0*
_output_shapes
: 2
cond/pruning_ops/mult
cond/pruning_ops/RoundRoundcond/pruning_ops/mul:z:0*
T0*
_output_shapes
: 2
cond/pruning_ops/Round?
cond/pruning_ops/Cast_1Castcond/pruning_ops/Round:y:0*

DstT0*

SrcT0*
_output_shapes
: 2
cond/pruning_ops/Cast_1?
cond/pruning_ops/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2 
cond/pruning_ops/Reshape/shape?
cond/pruning_ops/ReshapeReshapecond/pruning_ops/Abs:y:0'cond/pruning_ops/Reshape/shape:output:0*
T0*
_output_shapes	
:?2
cond/pruning_ops/Reshapeu
cond/pruning_ops/Size_1Const*
_output_shapes
: *
dtype0*
value
B :?2
cond/pruning_ops/Size_1?
cond/pruning_ops/TopKV2TopKV2!cond/pruning_ops/Reshape:output:0 cond/pruning_ops/Size_1:output:0*
T0*"
_output_shapes
:?:?2
cond/pruning_ops/TopKV2v
cond/pruning_ops/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
cond/pruning_ops/sub_1/y?
cond/pruning_ops/sub_1Subcond/pruning_ops/Cast_1:y:0!cond/pruning_ops/sub_1/y:output:0*
T0*
_output_shapes
: 2
cond/pruning_ops/sub_1?
cond/pruning_ops/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
cond/pruning_ops/GatherV2/axis?
cond/pruning_ops/GatherV2GatherV2 cond/pruning_ops/TopKV2:values:0cond/pruning_ops/sub_1:z:0'cond/pruning_ops/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: 2
cond/pruning_ops/GatherV2?
cond/pruning_ops/GreaterEqualGreaterEqualcond/pruning_ops/Abs:y:0"cond/pruning_ops/GatherV2:output:0*
T0*&
_output_shapes
:
2
cond/pruning_ops/GreaterEqual?
cond/pruning_ops/Cast_2Cast!cond/pruning_ops/GreaterEqual:z:0*

DstT0*

SrcT0
*&
_output_shapes
:
2
cond/pruning_ops/Cast_2?
cond/AssignVariableOpAssignVariableOpcond_assignvariableop_resourcecond/pruning_ops/Cast_2:y:0*
_output_shapes
 *
dtype02
cond/AssignVariableOp?
cond/AssignVariableOp_1AssignVariableOp cond_assignvariableop_1_resource"cond/pruning_ops/GatherV2:output:0*
_output_shapes
 *
dtype02
cond/AssignVariableOp_1?
cond/group_depsNoOp^cond/AssignVariableOp^cond/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2
cond/group_depsy
cond/IdentityIdentitycond_identity_logicaland_1^cond/group_deps*
T0
*
_output_shapes
: 2
cond/Identity?
cond/Identity_1Identitycond/Identity:output:0^cond/AssignVariableOp^cond/AssignVariableOp_1!^cond/GreaterEqual/ReadVariableOp^cond/LessEqual/ReadVariableOp^cond/Sub/ReadVariableOp$^cond/pruning_ops/Abs/ReadVariableOp*
T0
*
_output_shapes
: 2
cond/Identity_1"+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2.
cond/AssignVariableOpcond/AssignVariableOp22
cond/AssignVariableOp_1cond/AssignVariableOp_12D
 cond/GreaterEqual/ReadVariableOp cond/GreaterEqual/ReadVariableOp2>
cond/LessEqual/ReadVariableOpcond/LessEqual/ReadVariableOp22
cond/Sub/ReadVariableOpcond/Sub/ReadVariableOp2J
#cond/pruning_ops/Abs/ReadVariableOp#cond/pruning_ops/Abs/ReadVariableOp:

_output_shapes
: 
?	
?
3assert_greater_equal_Assert_AssertGuard_true_247822M
Iassert_greater_equal_assert_assertguard_identity_assert_greater_equal_all
7
3assert_greater_equal_assert_assertguard_placeholder	9
5assert_greater_equal_assert_assertguard_placeholder_1	6
2assert_greater_equal_assert_assertguard_identity_1
z
,assert_greater_equal/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2.
,assert_greater_equal/Assert/AssertGuard/NoOp?
0assert_greater_equal/Assert/AssertGuard/IdentityIdentityIassert_greater_equal_assert_assertguard_identity_assert_greater_equal_all-^assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identity?
2assert_greater_equal/Assert/AssertGuard/Identity_1Identity9assert_greater_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 24
2assert_greater_equal/Assert/AssertGuard/Identity_1"q
2assert_greater_equal_assert_assertguard_identity_1;assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?,
?
__inference_adapt_step_247504
iterator

iterator_1%
add_readvariableop_resource:	 %
readvariableop_resource:'
readvariableop_2_resource:??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_2?IteratorGetNext?ReadVariableOp?ReadVariableOp_1?ReadVariableOp_2?add/ReadVariableOp?
IteratorGetNextIteratorGetNextiterator*
_class
loc:@iterator*/
_output_shapes
:?????????(1*.
output_shapes
:?????????(1*
output_types
22
IteratorGetNext?
moments/mean/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2 
moments/mean/reduction_indices?
moments/meanMeanIteratorGetNext:components:0'moments/mean/reduction_indices:output:0*
T0*&
_output_shapes
:*
	keep_dims(2
moments/mean?
moments/StopGradientStopGradientmoments/mean:output:0*
T0*&
_output_shapes
:2
moments/StopGradient?
moments/SquaredDifferenceSquaredDifferenceIteratorGetNext:components:0moments/StopGradient:output:0*
T0*/
_output_shapes
:?????????(12
moments/SquaredDifference?
"moments/variance/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2$
"moments/variance/reduction_indices?
moments/varianceMeanmoments/SquaredDifference:z:0+moments/variance/reduction_indices:output:0*
T0*&
_output_shapes
:*
	keep_dims(2
moments/variance?
moments/SqueezeSqueezemoments/mean:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze?
moments/Squeeze_1Squeezemoments/variance:output:0*
T0*
_output_shapes
:*
squeeze_dims
 2
moments/Squeeze_1j
ShapeShapeIteratorGetNext:components:0*
T0*
_output_shapes
:*
out_type0	2
Shapey
GatherV2/indicesConst*
_output_shapes
:*
dtype0*!
valueB"          2
GatherV2/indices`
GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2
GatherV2/axis?
GatherV2GatherV2Shape:output:0GatherV2/indices:output:0GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
:2

GatherV2X
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstX
ProdProdGatherV2:output:0Const:output:0*
T0	*
_output_shapes
: 2
Prod|
add/ReadVariableOpReadVariableOpadd_readvariableop_resource*
_output_shapes
: *
dtype0	2
add/ReadVariableOp_
addAddV2Prod:output:0add/ReadVariableOp:value:0*
T0	*
_output_shapes
: 2
addS
CastCastProd:output:0*

DstT0*

SrcT0	*
_output_shapes
: 2
CastQ
Cast_1Castadd:z:0*

DstT0*

SrcT0	*
_output_shapes
: 2
Cast_1T
truedivRealDivCast:y:0
Cast_1:y:0*
T0*
_output_shapes
: 2	
truedivS
sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sub/xO
subSubsub/x:output:0truediv:z:0*
T0*
_output_shapes
: 2
subt
ReadVariableOpReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOpW
mulMulReadVariableOp:value:0sub:z:0*
T0*
_output_shapes
:2
mula
mul_1Mulmoments/Squeeze:output:0truediv:z:0*
T0*
_output_shapes
:2
mul_1P
add_1AddV2mul:z:0	mul_1:z:0*
T0*
_output_shapes
:2
add_1x
ReadVariableOp_1ReadVariableOpreadvariableop_resource*
_output_shapes
:*
dtype02
ReadVariableOp_1_
sub_1SubReadVariableOp_1:value:0	add_1:z:0*
T0*
_output_shapes
:2
sub_1S
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/yQ
powPow	sub_1:z:0pow/y:output:0*
T0*
_output_shapes
:2
powz
ReadVariableOp_2ReadVariableOpreadvariableop_2_resource*
_output_shapes
:*
dtype02
ReadVariableOp_2_
add_2AddV2ReadVariableOp_2:value:0pow:z:0*
T0*
_output_shapes
:2
add_2N
mul_2Mul	add_2:z:0sub:z:0*
T0*
_output_shapes
:2
mul_2_
sub_2Submoments/Squeeze:output:0	add_1:z:0*
T0*
_output_shapes
:2
sub_2W
pow_1/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2	
pow_1/yW
pow_1Pow	sub_2:z:0pow_1/y:output:0*
T0*
_output_shapes
:2
pow_1c
add_3AddV2moments/Squeeze_1:output:0	pow_1:z:0*
T0*
_output_shapes
:2
add_3R
mul_3Mul	add_3:z:0truediv:z:0*
T0*
_output_shapes
:2
mul_3R
add_4AddV2	mul_2:z:0	mul_3:z:0*
T0*
_output_shapes
:2
add_4?
AssignVariableOpAssignVariableOpreadvariableop_resource	add_1:z:0^ReadVariableOp^ReadVariableOp_1*
_output_shapes
 *
dtype02
AssignVariableOp?
AssignVariableOp_1AssignVariableOpreadvariableop_2_resource	add_4:z:0^ReadVariableOp_2*
_output_shapes
 *
dtype02
AssignVariableOp_1?
AssignVariableOp_2AssignVariableOpadd_readvariableop_resourceadd:z:0^add/ReadVariableOp*
_output_shapes
 *
dtype0	2
AssignVariableOp_2*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22"
IteratorGetNextIteratorGetNext2 
ReadVariableOpReadVariableOp2$
ReadVariableOp_1ReadVariableOp_12$
ReadVariableOp_2ReadVariableOp_22(
add/ReadVariableOpadd/ReadVariableOp:( $
"
_user_specified_name
iterator:@<

_output_shapes
: 
"
_user_specified_name
iterator
?
?
cond_false_247974
cond_placeholder
cond_placeholder_1
cond_placeholder_2
cond_placeholder_3
cond_identity_logicaland_1

cond_identity_1
4
	cond/NoOpNoOp*
_output_shapes
 2
	cond/NoOps
cond/IdentityIdentitycond_identity_logicaland_1
^cond/NoOp*
T0
*
_output_shapes
: 2
cond/Identityg
cond/Identity_1Identitycond/Identity:output:0*
T0
*
_output_shapes
: 2
cond/Identity_1"+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : :

_output_shapes
: 
?
?
cond_false_246283
cond_placeholder
cond_placeholder_1
cond_placeholder_2
cond_placeholder_3
cond_identity_logicaland_1

cond_identity_1
4
	cond/NoOpNoOp*
_output_shapes
 2
	cond/NoOps
cond/IdentityIdentitycond_identity_logicaland_1
^cond/NoOp*
T0
*
_output_shapes
: 2
cond/Identityg
cond/Identity_1Identitycond/Identity:output:0*
T0
*
_output_shapes
: 2
cond/Identity_1"+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : :

_output_shapes
: 
?
a
(__inference_dropout_layer_call_fn_247771

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
GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_2464712
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
?	
?
3assert_greater_equal_Assert_AssertGuard_true_246391M
Iassert_greater_equal_assert_assertguard_identity_assert_greater_equal_all
7
3assert_greater_equal_assert_assertguard_placeholder	9
5assert_greater_equal_assert_assertguard_placeholder_1	6
2assert_greater_equal_assert_assertguard_identity_1
z
,assert_greater_equal/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2.
,assert_greater_equal/Assert/AssertGuard/NoOp?
0assert_greater_equal/Assert/AssertGuard/IdentityIdentityIassert_greater_equal_assert_assertguard_identity_assert_greater_equal_all-^assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identity?
2assert_greater_equal/Assert/AssertGuard/Identity_1Identity9assert_greater_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 24
2assert_greater_equal/Assert/AssertGuard/Identity_1"q
2assert_greater_equal_assert_assertguard_identity_1;assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
y
]__inference_prune_low_magnitude_max_pooling2d_layer_call_and_return_conditional_losses_247691

inputs
identity4
	no_updateNoOp*
_output_shapes
 2
	no_update6

group_depsNoOp*
_output_shapes
 2

group_deps?
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
?z
?
!__inference__wrapped_model_246093
input_1Q
Cextaudiodatamodelprun_normalization_reshape_readvariableop_resource:S
Eextaudiodatamodelprun_normalization_reshape_1_readvariableop_resource:f
Lextaudiodatamodelprun_prune_low_magnitude_conv2d_mul_readvariableop_resource:
h
Nextaudiodatamodelprun_prune_low_magnitude_conv2d_mul_readvariableop_1_resource:
^
Pextaudiodatamodelprun_prune_low_magnitude_conv2d_biasadd_readvariableop_resource:^
Kextaudiodatamodelprun_prune_low_magnitude_dense_mul_readvariableop_resource:	?`
Mextaudiodatamodelprun_prune_low_magnitude_dense_mul_readvariableop_1_resource:	?]
Oextaudiodatamodelprun_prune_low_magnitude_dense_biasadd_readvariableop_resource:
identity??:ExtAudioDataModelPrun/normalization/Reshape/ReadVariableOp?<ExtAudioDataModelPrun/normalization/Reshape_1/ReadVariableOp?AExtAudioDataModelPrun/prune_low_magnitude_conv2d/AssignVariableOp?GExtAudioDataModelPrun/prune_low_magnitude_conv2d/BiasAdd/ReadVariableOp?FExtAudioDataModelPrun/prune_low_magnitude_conv2d/Conv2D/ReadVariableOp?CExtAudioDataModelPrun/prune_low_magnitude_conv2d/Mul/ReadVariableOp?EExtAudioDataModelPrun/prune_low_magnitude_conv2d/Mul/ReadVariableOp_1?@ExtAudioDataModelPrun/prune_low_magnitude_dense/AssignVariableOp?FExtAudioDataModelPrun/prune_low_magnitude_dense/BiasAdd/ReadVariableOp?EExtAudioDataModelPrun/prune_low_magnitude_dense/MatMul/ReadVariableOp?BExtAudioDataModelPrun/prune_low_magnitude_dense/Mul/ReadVariableOp?DExtAudioDataModelPrun/prune_low_magnitude_dense/Mul/ReadVariableOp_1?
:ExtAudioDataModelPrun/normalization/Reshape/ReadVariableOpReadVariableOpCextaudiodatamodelprun_normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02<
:ExtAudioDataModelPrun/normalization/Reshape/ReadVariableOp?
1ExtAudioDataModelPrun/normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            23
1ExtAudioDataModelPrun/normalization/Reshape/shape?
+ExtAudioDataModelPrun/normalization/ReshapeReshapeBExtAudioDataModelPrun/normalization/Reshape/ReadVariableOp:value:0:ExtAudioDataModelPrun/normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:2-
+ExtAudioDataModelPrun/normalization/Reshape?
<ExtAudioDataModelPrun/normalization/Reshape_1/ReadVariableOpReadVariableOpEextaudiodatamodelprun_normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02>
<ExtAudioDataModelPrun/normalization/Reshape_1/ReadVariableOp?
3ExtAudioDataModelPrun/normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            25
3ExtAudioDataModelPrun/normalization/Reshape_1/shape?
-ExtAudioDataModelPrun/normalization/Reshape_1ReshapeDExtAudioDataModelPrun/normalization/Reshape_1/ReadVariableOp:value:0<ExtAudioDataModelPrun/normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2/
-ExtAudioDataModelPrun/normalization/Reshape_1?
'ExtAudioDataModelPrun/normalization/subSubinput_14ExtAudioDataModelPrun/normalization/Reshape:output:0*
T0*/
_output_shapes
:?????????(12)
'ExtAudioDataModelPrun/normalization/sub?
(ExtAudioDataModelPrun/normalization/SqrtSqrt6ExtAudioDataModelPrun/normalization/Reshape_1:output:0*
T0*&
_output_shapes
:2*
(ExtAudioDataModelPrun/normalization/Sqrt?
-ExtAudioDataModelPrun/normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32/
-ExtAudioDataModelPrun/normalization/Maximum/y?
+ExtAudioDataModelPrun/normalization/MaximumMaximum,ExtAudioDataModelPrun/normalization/Sqrt:y:06ExtAudioDataModelPrun/normalization/Maximum/y:output:0*
T0*&
_output_shapes
:2-
+ExtAudioDataModelPrun/normalization/Maximum?
+ExtAudioDataModelPrun/normalization/truedivRealDiv+ExtAudioDataModelPrun/normalization/sub:z:0/ExtAudioDataModelPrun/normalization/Maximum:z:0*
T0*/
_output_shapes
:?????????(12-
+ExtAudioDataModelPrun/normalization/truediv?
:ExtAudioDataModelPrun/prune_low_magnitude_conv2d/no_updateNoOp*
_output_shapes
 2<
:ExtAudioDataModelPrun/prune_low_magnitude_conv2d/no_update?
CExtAudioDataModelPrun/prune_low_magnitude_conv2d/Mul/ReadVariableOpReadVariableOpLextaudiodatamodelprun_prune_low_magnitude_conv2d_mul_readvariableop_resource*&
_output_shapes
:
*
dtype02E
CExtAudioDataModelPrun/prune_low_magnitude_conv2d/Mul/ReadVariableOp?
EExtAudioDataModelPrun/prune_low_magnitude_conv2d/Mul/ReadVariableOp_1ReadVariableOpNextaudiodatamodelprun_prune_low_magnitude_conv2d_mul_readvariableop_1_resource*&
_output_shapes
:
*
dtype02G
EExtAudioDataModelPrun/prune_low_magnitude_conv2d/Mul/ReadVariableOp_1?
4ExtAudioDataModelPrun/prune_low_magnitude_conv2d/MulMulKExtAudioDataModelPrun/prune_low_magnitude_conv2d/Mul/ReadVariableOp:value:0MExtAudioDataModelPrun/prune_low_magnitude_conv2d/Mul/ReadVariableOp_1:value:0*
T0*&
_output_shapes
:
26
4ExtAudioDataModelPrun/prune_low_magnitude_conv2d/Mul?
AExtAudioDataModelPrun/prune_low_magnitude_conv2d/AssignVariableOpAssignVariableOpLextaudiodatamodelprun_prune_low_magnitude_conv2d_mul_readvariableop_resource8ExtAudioDataModelPrun/prune_low_magnitude_conv2d/Mul:z:0D^ExtAudioDataModelPrun/prune_low_magnitude_conv2d/Mul/ReadVariableOp*
_output_shapes
 *
dtype02C
AExtAudioDataModelPrun/prune_low_magnitude_conv2d/AssignVariableOp?
;ExtAudioDataModelPrun/prune_low_magnitude_conv2d/group_depsNoOpB^ExtAudioDataModelPrun/prune_low_magnitude_conv2d/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2=
;ExtAudioDataModelPrun/prune_low_magnitude_conv2d/group_deps?
=ExtAudioDataModelPrun/prune_low_magnitude_conv2d/group_deps_1NoOp<^ExtAudioDataModelPrun/prune_low_magnitude_conv2d/group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2?
=ExtAudioDataModelPrun/prune_low_magnitude_conv2d/group_deps_1?
FExtAudioDataModelPrun/prune_low_magnitude_conv2d/Conv2D/ReadVariableOpReadVariableOpLextaudiodatamodelprun_prune_low_magnitude_conv2d_mul_readvariableop_resourceB^ExtAudioDataModelPrun/prune_low_magnitude_conv2d/AssignVariableOp*&
_output_shapes
:
*
dtype02H
FExtAudioDataModelPrun/prune_low_magnitude_conv2d/Conv2D/ReadVariableOp?
7ExtAudioDataModelPrun/prune_low_magnitude_conv2d/Conv2DConv2D/ExtAudioDataModelPrun/normalization/truediv:z:0NExtAudioDataModelPrun/prune_low_magnitude_conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
29
7ExtAudioDataModelPrun/prune_low_magnitude_conv2d/Conv2D?
GExtAudioDataModelPrun/prune_low_magnitude_conv2d/BiasAdd/ReadVariableOpReadVariableOpPextaudiodatamodelprun_prune_low_magnitude_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02I
GExtAudioDataModelPrun/prune_low_magnitude_conv2d/BiasAdd/ReadVariableOp?
8ExtAudioDataModelPrun/prune_low_magnitude_conv2d/BiasAddBiasAdd@ExtAudioDataModelPrun/prune_low_magnitude_conv2d/Conv2D:output:0OExtAudioDataModelPrun/prune_low_magnitude_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2:
8ExtAudioDataModelPrun/prune_low_magnitude_conv2d/BiasAdd?
6ExtAudioDataModelPrun/prune_low_magnitude_conv2d/Relu6Relu6AExtAudioDataModelPrun/prune_low_magnitude_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????28
6ExtAudioDataModelPrun/prune_low_magnitude_conv2d/Relu6?
AExtAudioDataModelPrun/prune_low_magnitude_max_pooling2d/no_updateNoOp*
_output_shapes
 2C
AExtAudioDataModelPrun/prune_low_magnitude_max_pooling2d/no_update?
BExtAudioDataModelPrun/prune_low_magnitude_max_pooling2d/group_depsNoOp*
_output_shapes
 2D
BExtAudioDataModelPrun/prune_low_magnitude_max_pooling2d/group_deps?
?ExtAudioDataModelPrun/prune_low_magnitude_max_pooling2d/MaxPoolMaxPoolDExtAudioDataModelPrun/prune_low_magnitude_conv2d/Relu6:activations:0*/
_output_shapes
:?????????
*
ksize
*
paddingVALID*
strides
2A
?ExtAudioDataModelPrun/prune_low_magnitude_max_pooling2d/MaxPool?
&ExtAudioDataModelPrun/dropout/IdentityIdentityHExtAudioDataModelPrun/prune_low_magnitude_max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:?????????
2(
&ExtAudioDataModelPrun/dropout/Identity?
;ExtAudioDataModelPrun/prune_low_magnitude_flatten/no_updateNoOp*
_output_shapes
 2=
;ExtAudioDataModelPrun/prune_low_magnitude_flatten/no_update?
<ExtAudioDataModelPrun/prune_low_magnitude_flatten/group_depsNoOp*
_output_shapes
 2>
<ExtAudioDataModelPrun/prune_low_magnitude_flatten/group_deps?
7ExtAudioDataModelPrun/prune_low_magnitude_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  29
7ExtAudioDataModelPrun/prune_low_magnitude_flatten/Const?
9ExtAudioDataModelPrun/prune_low_magnitude_flatten/ReshapeReshape/ExtAudioDataModelPrun/dropout/Identity:output:0@ExtAudioDataModelPrun/prune_low_magnitude_flatten/Const:output:0*
T0*(
_output_shapes
:??????????2;
9ExtAudioDataModelPrun/prune_low_magnitude_flatten/Reshape?
9ExtAudioDataModelPrun/prune_low_magnitude_dense/no_updateNoOp*
_output_shapes
 2;
9ExtAudioDataModelPrun/prune_low_magnitude_dense/no_update?
BExtAudioDataModelPrun/prune_low_magnitude_dense/Mul/ReadVariableOpReadVariableOpKextaudiodatamodelprun_prune_low_magnitude_dense_mul_readvariableop_resource*
_output_shapes
:	?*
dtype02D
BExtAudioDataModelPrun/prune_low_magnitude_dense/Mul/ReadVariableOp?
DExtAudioDataModelPrun/prune_low_magnitude_dense/Mul/ReadVariableOp_1ReadVariableOpMextaudiodatamodelprun_prune_low_magnitude_dense_mul_readvariableop_1_resource*
_output_shapes
:	?*
dtype02F
DExtAudioDataModelPrun/prune_low_magnitude_dense/Mul/ReadVariableOp_1?
3ExtAudioDataModelPrun/prune_low_magnitude_dense/MulMulJExtAudioDataModelPrun/prune_low_magnitude_dense/Mul/ReadVariableOp:value:0LExtAudioDataModelPrun/prune_low_magnitude_dense/Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	?25
3ExtAudioDataModelPrun/prune_low_magnitude_dense/Mul?
@ExtAudioDataModelPrun/prune_low_magnitude_dense/AssignVariableOpAssignVariableOpKextaudiodatamodelprun_prune_low_magnitude_dense_mul_readvariableop_resource7ExtAudioDataModelPrun/prune_low_magnitude_dense/Mul:z:0C^ExtAudioDataModelPrun/prune_low_magnitude_dense/Mul/ReadVariableOp*
_output_shapes
 *
dtype02B
@ExtAudioDataModelPrun/prune_low_magnitude_dense/AssignVariableOp?
:ExtAudioDataModelPrun/prune_low_magnitude_dense/group_depsNoOpA^ExtAudioDataModelPrun/prune_low_magnitude_dense/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2<
:ExtAudioDataModelPrun/prune_low_magnitude_dense/group_deps?
<ExtAudioDataModelPrun/prune_low_magnitude_dense/group_deps_1NoOp;^ExtAudioDataModelPrun/prune_low_magnitude_dense/group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2>
<ExtAudioDataModelPrun/prune_low_magnitude_dense/group_deps_1?
EExtAudioDataModelPrun/prune_low_magnitude_dense/MatMul/ReadVariableOpReadVariableOpKextaudiodatamodelprun_prune_low_magnitude_dense_mul_readvariableop_resourceA^ExtAudioDataModelPrun/prune_low_magnitude_dense/AssignVariableOp*
_output_shapes
:	?*
dtype02G
EExtAudioDataModelPrun/prune_low_magnitude_dense/MatMul/ReadVariableOp?
6ExtAudioDataModelPrun/prune_low_magnitude_dense/MatMulMatMulBExtAudioDataModelPrun/prune_low_magnitude_flatten/Reshape:output:0MExtAudioDataModelPrun/prune_low_magnitude_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????28
6ExtAudioDataModelPrun/prune_low_magnitude_dense/MatMul?
FExtAudioDataModelPrun/prune_low_magnitude_dense/BiasAdd/ReadVariableOpReadVariableOpOextaudiodatamodelprun_prune_low_magnitude_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02H
FExtAudioDataModelPrun/prune_low_magnitude_dense/BiasAdd/ReadVariableOp?
7ExtAudioDataModelPrun/prune_low_magnitude_dense/BiasAddBiasAdd@ExtAudioDataModelPrun/prune_low_magnitude_dense/MatMul:product:0NExtAudioDataModelPrun/prune_low_magnitude_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????29
7ExtAudioDataModelPrun/prune_low_magnitude_dense/BiasAdd?
IdentityIdentity@ExtAudioDataModelPrun/prune_low_magnitude_dense/BiasAdd:output:0;^ExtAudioDataModelPrun/normalization/Reshape/ReadVariableOp=^ExtAudioDataModelPrun/normalization/Reshape_1/ReadVariableOpB^ExtAudioDataModelPrun/prune_low_magnitude_conv2d/AssignVariableOpH^ExtAudioDataModelPrun/prune_low_magnitude_conv2d/BiasAdd/ReadVariableOpG^ExtAudioDataModelPrun/prune_low_magnitude_conv2d/Conv2D/ReadVariableOpD^ExtAudioDataModelPrun/prune_low_magnitude_conv2d/Mul/ReadVariableOpF^ExtAudioDataModelPrun/prune_low_magnitude_conv2d/Mul/ReadVariableOp_1A^ExtAudioDataModelPrun/prune_low_magnitude_dense/AssignVariableOpG^ExtAudioDataModelPrun/prune_low_magnitude_dense/BiasAdd/ReadVariableOpF^ExtAudioDataModelPrun/prune_low_magnitude_dense/MatMul/ReadVariableOpC^ExtAudioDataModelPrun/prune_low_magnitude_dense/Mul/ReadVariableOpE^ExtAudioDataModelPrun/prune_low_magnitude_dense/Mul/ReadVariableOp_1*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????(1: : : : : : : : 2x
:ExtAudioDataModelPrun/normalization/Reshape/ReadVariableOp:ExtAudioDataModelPrun/normalization/Reshape/ReadVariableOp2|
<ExtAudioDataModelPrun/normalization/Reshape_1/ReadVariableOp<ExtAudioDataModelPrun/normalization/Reshape_1/ReadVariableOp2?
AExtAudioDataModelPrun/prune_low_magnitude_conv2d/AssignVariableOpAExtAudioDataModelPrun/prune_low_magnitude_conv2d/AssignVariableOp2?
GExtAudioDataModelPrun/prune_low_magnitude_conv2d/BiasAdd/ReadVariableOpGExtAudioDataModelPrun/prune_low_magnitude_conv2d/BiasAdd/ReadVariableOp2?
FExtAudioDataModelPrun/prune_low_magnitude_conv2d/Conv2D/ReadVariableOpFExtAudioDataModelPrun/prune_low_magnitude_conv2d/Conv2D/ReadVariableOp2?
CExtAudioDataModelPrun/prune_low_magnitude_conv2d/Mul/ReadVariableOpCExtAudioDataModelPrun/prune_low_magnitude_conv2d/Mul/ReadVariableOp2?
EExtAudioDataModelPrun/prune_low_magnitude_conv2d/Mul/ReadVariableOp_1EExtAudioDataModelPrun/prune_low_magnitude_conv2d/Mul/ReadVariableOp_12?
@ExtAudioDataModelPrun/prune_low_magnitude_dense/AssignVariableOp@ExtAudioDataModelPrun/prune_low_magnitude_dense/AssignVariableOp2?
FExtAudioDataModelPrun/prune_low_magnitude_dense/BiasAdd/ReadVariableOpFExtAudioDataModelPrun/prune_low_magnitude_dense/BiasAdd/ReadVariableOp2?
EExtAudioDataModelPrun/prune_low_magnitude_dense/MatMul/ReadVariableOpEExtAudioDataModelPrun/prune_low_magnitude_dense/MatMul/ReadVariableOp2?
BExtAudioDataModelPrun/prune_low_magnitude_dense/Mul/ReadVariableOpBExtAudioDataModelPrun/prune_low_magnitude_dense/Mul/ReadVariableOp2?
DExtAudioDataModelPrun/prune_low_magnitude_dense/Mul/ReadVariableOp_1DExtAudioDataModelPrun/prune_low_magnitude_dense/Mul/ReadVariableOp_1:X T
/
_output_shapes
:?????????(1
!
_user_specified_name	input_1
?
y
]__inference_prune_low_magnitude_max_pooling2d_layer_call_and_return_conditional_losses_246153

inputs
identity4
	no_updateNoOp*
_output_shapes
 2
	no_update6

group_depsNoOp*
_output_shapes
 2

group_deps?
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
?I
?
V__inference_prune_low_magnitude_conv2d_layer_call_and_return_conditional_losses_246700

inputs6
,assert_greater_equal_readvariableop_resource:	 &
cond_input_1:
&
cond_input_2:

cond_input_3: -
biasadd_readvariableop_resource:
identity??AssignVariableOp?BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?GreaterEqual/ReadVariableOp?LessEqual/ReadVariableOp?Mul/ReadVariableOp?Mul/ReadVariableOp_1?Sub/ReadVariableOp?'assert_greater_equal/Assert/AssertGuard?#assert_greater_equal/ReadVariableOp?cond?
#assert_greater_equal/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource*
_output_shapes
: *
dtype0	2%
#assert_greater_equal/ReadVariableOpr
assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
assert_greater_equal/y?
!assert_greater_equal/GreaterEqualGreaterEqual+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
T0	*
_output_shapes
: 2#
!assert_greater_equal/GreaterEqualx
assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 2
assert_greater_equal/Rank?
 assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2"
 assert_greater_equal/range/start?
 assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2"
 assert_greater_equal/range/delta?
assert_greater_equal/rangeRange)assert_greater_equal/range/start:output:0"assert_greater_equal/Rank:output:0)assert_greater_equal/range/delta:output:0*
_output_shapes
: 2
assert_greater_equal/range?
assert_greater_equal/AllAll%assert_greater_equal/GreaterEqual:z:0#assert_greater_equal/range:output:0*
_output_shapes
: 2
assert_greater_equal/All?
!assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*?
value?B? B?Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2#
!assert_greater_equal/Assert/Const?
#assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2%
#assert_greater_equal/Assert/Const_1?
#assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2%
#assert_greater_equal/Assert/Const_2?
#assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2%
#assert_greater_equal/Assert/Const_3?
'assert_greater_equal/Assert/AssertGuardIf!assert_greater_equal/All:output:0!assert_greater_equal/All:output:0+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *G
else_branch8R6
4assert_greater_equal_Assert_AssertGuard_false_246588*
output_shapes
: *F
then_branch7R5
3assert_greater_equal_Assert_AssertGuard_true_2465872)
'assert_greater_equal/Assert/AssertGuard?
0assert_greater_equal/Assert/AssertGuard/IdentityIdentity0assert_greater_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identity?
GreaterEqual/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOp?
GreaterEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y?
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqual?
LessEqual/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp?
LessEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqual?
Less/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB :
?????????2
Less/x?
Less/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd?
Sub/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOp?
Sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
Sub?

FloorMod/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorMod?
Equal/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1?
ConstConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Const?
condIfLogicalAnd_1:z:0,assert_greater_equal_readvariableop_resourcecond_input_1cond_input_2cond_input_3LogicalAnd_1:z:0*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: *$
_read_only_resource_inputs
*$
else_branchR
cond_false_246628*
output_shapes
: *#
then_branchR
cond_true_2466272
condZ
cond/IdentityIdentitycond:output:0*
T0
*
_output_shapes
: 2
cond/Identityq
updateNoOp1^assert_greater_equal/Assert/AssertGuard/Identity^cond/Identity*
_output_shapes
 2
update}
Mul/ReadVariableOpReadVariableOpcond_input_1*&
_output_shapes
:
*
dtype02
Mul/ReadVariableOp?
Mul/ReadVariableOp_1ReadVariableOpcond_input_2^cond*&
_output_shapes
:
*
dtype02
Mul/ReadVariableOp_1|
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*&
_output_shapes
:
2
Mul?
AssignVariableOpAssignVariableOpcond_input_1Mul:z:0^Mul/ReadVariableOp^cond*
_output_shapes
 *
dtype02
AssignVariableOpw

group_depsNoOp^AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2

group_depsu
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2
group_deps_1?
Conv2D/ReadVariableOpReadVariableOpcond_input_1^AssignVariableOp*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
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
BiasAddc
Relu6Relu6BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu6?
IdentityIdentityRelu6:activations:0^AssignVariableOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp^GreaterEqual/ReadVariableOp^LessEqual/ReadVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1^Sub/ReadVariableOp(^assert_greater_equal/Assert/AssertGuard$^assert_greater_equal/ReadVariableOp^cond*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????(1: : : : : 2$
AssignVariableOpAssignVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2:
GreaterEqual/ReadVariableOpGreaterEqual/ReadVariableOp24
LessEqual/ReadVariableOpLessEqual/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_12(
Sub/ReadVariableOpSub/ReadVariableOp2R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard2J
#assert_greater_equal/ReadVariableOp#assert_greater_equal/ReadVariableOp2
condcond:W S
/
_output_shapes
:?????????(1
 
_user_specified_nameinputs
?
?
4assert_greater_equal_Assert_AssertGuard_false_247562K
Gassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all
V
Rassert_greater_equal_assert_assertguard_assert_assert_greater_equal_readvariableop	I
Eassert_greater_equal_assert_assertguard_assert_assert_greater_equal_y	6
2assert_greater_equal_assert_assertguard_identity_1
??.assert_greater_equal/Assert/AssertGuard/Assert?
5assert_greater_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*?
value?B? B?Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.27
5assert_greater_equal/Assert/AssertGuard/Assert/data_0?
5assert_greater_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:27
5assert_greater_equal/Assert/AssertGuard/Assert/data_1?
5assert_greater_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 27
5assert_greater_equal/Assert/AssertGuard/Assert/data_2?
5assert_greater_equal/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 27
5assert_greater_equal/Assert/AssertGuard/Assert/data_4?
.assert_greater_equal/Assert/AssertGuard/AssertAssertGassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all>assert_greater_equal/Assert/AssertGuard/Assert/data_0:output:0>assert_greater_equal/Assert/AssertGuard/Assert/data_1:output:0>assert_greater_equal/Assert/AssertGuard/Assert/data_2:output:0Rassert_greater_equal_assert_assertguard_assert_assert_greater_equal_readvariableop>assert_greater_equal/Assert/AssertGuard/Assert/data_4:output:0Eassert_greater_equal_assert_assertguard_assert_assert_greater_equal_y*
T

2		*
_output_shapes
 20
.assert_greater_equal/Assert/AssertGuard/Assert?
0assert_greater_equal/Assert/AssertGuard/IdentityIdentityGassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all/^assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identity?
2assert_greater_equal/Assert/AssertGuard/Identity_1Identity9assert_greater_equal/Assert/AssertGuard/Identity:output:0/^assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 24
2assert_greater_equal/Assert/AssertGuard/Identity_1"q
2assert_greater_equal_assert_assertguard_identity_1;assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2`
.assert_greater_equal/Assert/AssertGuard/Assert.assert_greater_equal/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
:__inference_prune_low_magnitude_dense_layer_call_fn_247903

inputs
unknown:	 
	unknown_0:	?
	unknown_1:	?
	unknown_2: 
	unknown_3:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_prune_low_magnitude_dense_layer_call_and_return_conditional_losses_2463542
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
4assert_greater_equal_Assert_AssertGuard_false_247823K
Gassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all
V
Rassert_greater_equal_assert_assertguard_assert_assert_greater_equal_readvariableop	I
Eassert_greater_equal_assert_assertguard_assert_assert_greater_equal_y	6
2assert_greater_equal_assert_assertguard_identity_1
??.assert_greater_equal/Assert/AssertGuard/Assert?
5assert_greater_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*?
value?B? B?Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.27
5assert_greater_equal/Assert/AssertGuard/Assert/data_0?
5assert_greater_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:27
5assert_greater_equal/Assert/AssertGuard/Assert/data_1?
5assert_greater_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 27
5assert_greater_equal/Assert/AssertGuard/Assert/data_2?
5assert_greater_equal/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 27
5assert_greater_equal/Assert/AssertGuard/Assert/data_4?
.assert_greater_equal/Assert/AssertGuard/AssertAssertGassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all>assert_greater_equal/Assert/AssertGuard/Assert/data_0:output:0>assert_greater_equal/Assert/AssertGuard/Assert/data_1:output:0>assert_greater_equal/Assert/AssertGuard/Assert/data_2:output:0Rassert_greater_equal_assert_assertguard_assert_assert_greater_equal_readvariableop>assert_greater_equal/Assert/AssertGuard/Assert/data_4:output:0Eassert_greater_equal_assert_assertguard_assert_assert_greater_equal_y*
T

2		*
_output_shapes
 20
.assert_greater_equal/Assert/AssertGuard/Assert?
0assert_greater_equal/Assert/AssertGuard/IdentityIdentityGassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all/^assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identity?
2assert_greater_equal/Assert/AssertGuard/Identity_1Identity9assert_greater_equal/Assert/AssertGuard/Identity:output:0/^assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 24
2assert_greater_equal/Assert/AssertGuard/Identity_1"q
2assert_greater_equal_assert_assertguard_identity_1;assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2`
.assert_greater_equal/Assert/AssertGuard/Assert.assert_greater_equal/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
6__inference_ExtAudioDataModelPrun_layer_call_fn_246212
input_1
unknown:
	unknown_0:#
	unknown_1:
#
	unknown_2:

	unknown_3:
	unknown_4:	?
	unknown_5:	?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_ExtAudioDataModelPrun_layer_call_and_return_conditional_losses_2461932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????(1: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????(1
!
_user_specified_name	input_1
?
?
:__inference_prune_low_magnitude_dense_layer_call_fn_247888

inputs
unknown:	?
	unknown_0:	?
	unknown_1:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_prune_low_magnitude_dense_layer_call_and_return_conditional_losses_2461842
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:??????????: : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
J
.__inference_max_pooling2d_layer_call_fn_246105

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
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_2460992
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
 
_user_specified_nameinputs
?Y
?	
Q__inference_ExtAudioDataModelPrun_layer_call_and_return_conditional_losses_247051

inputs;
-normalization_reshape_readvariableop_resource:=
/normalization_reshape_1_readvariableop_resource:P
6prune_low_magnitude_conv2d_mul_readvariableop_resource:
R
8prune_low_magnitude_conv2d_mul_readvariableop_1_resource:
H
:prune_low_magnitude_conv2d_biasadd_readvariableop_resource:H
5prune_low_magnitude_dense_mul_readvariableop_resource:	?J
7prune_low_magnitude_dense_mul_readvariableop_1_resource:	?G
9prune_low_magnitude_dense_biasadd_readvariableop_resource:
identity??$normalization/Reshape/ReadVariableOp?&normalization/Reshape_1/ReadVariableOp?+prune_low_magnitude_conv2d/AssignVariableOp?1prune_low_magnitude_conv2d/BiasAdd/ReadVariableOp?0prune_low_magnitude_conv2d/Conv2D/ReadVariableOp?-prune_low_magnitude_conv2d/Mul/ReadVariableOp?/prune_low_magnitude_conv2d/Mul/ReadVariableOp_1?*prune_low_magnitude_dense/AssignVariableOp?0prune_low_magnitude_dense/BiasAdd/ReadVariableOp?/prune_low_magnitude_dense/MatMul/ReadVariableOp?,prune_low_magnitude_dense/Mul/ReadVariableOp?.prune_low_magnitude_dense/Mul/ReadVariableOp_1?
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp?
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape/shape?
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:2
normalization/Reshape?
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp?
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape_1/shape?
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
normalization/Reshape_1?
normalization/subSubinputsnormalization/Reshape:output:0*
T0*/
_output_shapes
:?????????(12
normalization/sub?
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*&
_output_shapes
:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*&
_output_shapes
:2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*/
_output_shapes
:?????????(12
normalization/truedivj
$prune_low_magnitude_conv2d/no_updateNoOp*
_output_shapes
 2&
$prune_low_magnitude_conv2d/no_update?
-prune_low_magnitude_conv2d/Mul/ReadVariableOpReadVariableOp6prune_low_magnitude_conv2d_mul_readvariableop_resource*&
_output_shapes
:
*
dtype02/
-prune_low_magnitude_conv2d/Mul/ReadVariableOp?
/prune_low_magnitude_conv2d/Mul/ReadVariableOp_1ReadVariableOp8prune_low_magnitude_conv2d_mul_readvariableop_1_resource*&
_output_shapes
:
*
dtype021
/prune_low_magnitude_conv2d/Mul/ReadVariableOp_1?
prune_low_magnitude_conv2d/MulMul5prune_low_magnitude_conv2d/Mul/ReadVariableOp:value:07prune_low_magnitude_conv2d/Mul/ReadVariableOp_1:value:0*
T0*&
_output_shapes
:
2 
prune_low_magnitude_conv2d/Mul?
+prune_low_magnitude_conv2d/AssignVariableOpAssignVariableOp6prune_low_magnitude_conv2d_mul_readvariableop_resource"prune_low_magnitude_conv2d/Mul:z:0.^prune_low_magnitude_conv2d/Mul/ReadVariableOp*
_output_shapes
 *
dtype02-
+prune_low_magnitude_conv2d/AssignVariableOp?
%prune_low_magnitude_conv2d/group_depsNoOp,^prune_low_magnitude_conv2d/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2'
%prune_low_magnitude_conv2d/group_deps?
'prune_low_magnitude_conv2d/group_deps_1NoOp&^prune_low_magnitude_conv2d/group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2)
'prune_low_magnitude_conv2d/group_deps_1?
0prune_low_magnitude_conv2d/Conv2D/ReadVariableOpReadVariableOp6prune_low_magnitude_conv2d_mul_readvariableop_resource,^prune_low_magnitude_conv2d/AssignVariableOp*&
_output_shapes
:
*
dtype022
0prune_low_magnitude_conv2d/Conv2D/ReadVariableOp?
!prune_low_magnitude_conv2d/Conv2DConv2Dnormalization/truediv:z:08prune_low_magnitude_conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2#
!prune_low_magnitude_conv2d/Conv2D?
1prune_low_magnitude_conv2d/BiasAdd/ReadVariableOpReadVariableOp:prune_low_magnitude_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1prune_low_magnitude_conv2d/BiasAdd/ReadVariableOp?
"prune_low_magnitude_conv2d/BiasAddBiasAdd*prune_low_magnitude_conv2d/Conv2D:output:09prune_low_magnitude_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2$
"prune_low_magnitude_conv2d/BiasAdd?
 prune_low_magnitude_conv2d/Relu6Relu6+prune_low_magnitude_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2"
 prune_low_magnitude_conv2d/Relu6x
+prune_low_magnitude_max_pooling2d/no_updateNoOp*
_output_shapes
 2-
+prune_low_magnitude_max_pooling2d/no_updatez
,prune_low_magnitude_max_pooling2d/group_depsNoOp*
_output_shapes
 2.
,prune_low_magnitude_max_pooling2d/group_deps?
)prune_low_magnitude_max_pooling2d/MaxPoolMaxPool.prune_low_magnitude_conv2d/Relu6:activations:0*/
_output_shapes
:?????????
*
ksize
*
paddingVALID*
strides
2+
)prune_low_magnitude_max_pooling2d/MaxPool?
dropout/IdentityIdentity2prune_low_magnitude_max_pooling2d/MaxPool:output:0*
T0*/
_output_shapes
:?????????
2
dropout/Identityl
%prune_low_magnitude_flatten/no_updateNoOp*
_output_shapes
 2'
%prune_low_magnitude_flatten/no_updaten
&prune_low_magnitude_flatten/group_depsNoOp*
_output_shapes
 2(
&prune_low_magnitude_flatten/group_deps?
!prune_low_magnitude_flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"?????  2#
!prune_low_magnitude_flatten/Const?
#prune_low_magnitude_flatten/ReshapeReshapedropout/Identity:output:0*prune_low_magnitude_flatten/Const:output:0*
T0*(
_output_shapes
:??????????2%
#prune_low_magnitude_flatten/Reshapeh
#prune_low_magnitude_dense/no_updateNoOp*
_output_shapes
 2%
#prune_low_magnitude_dense/no_update?
,prune_low_magnitude_dense/Mul/ReadVariableOpReadVariableOp5prune_low_magnitude_dense_mul_readvariableop_resource*
_output_shapes
:	?*
dtype02.
,prune_low_magnitude_dense/Mul/ReadVariableOp?
.prune_low_magnitude_dense/Mul/ReadVariableOp_1ReadVariableOp7prune_low_magnitude_dense_mul_readvariableop_1_resource*
_output_shapes
:	?*
dtype020
.prune_low_magnitude_dense/Mul/ReadVariableOp_1?
prune_low_magnitude_dense/MulMul4prune_low_magnitude_dense/Mul/ReadVariableOp:value:06prune_low_magnitude_dense/Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	?2
prune_low_magnitude_dense/Mul?
*prune_low_magnitude_dense/AssignVariableOpAssignVariableOp5prune_low_magnitude_dense_mul_readvariableop_resource!prune_low_magnitude_dense/Mul:z:0-^prune_low_magnitude_dense/Mul/ReadVariableOp*
_output_shapes
 *
dtype02,
*prune_low_magnitude_dense/AssignVariableOp?
$prune_low_magnitude_dense/group_depsNoOp+^prune_low_magnitude_dense/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2&
$prune_low_magnitude_dense/group_deps?
&prune_low_magnitude_dense/group_deps_1NoOp%^prune_low_magnitude_dense/group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2(
&prune_low_magnitude_dense/group_deps_1?
/prune_low_magnitude_dense/MatMul/ReadVariableOpReadVariableOp5prune_low_magnitude_dense_mul_readvariableop_resource+^prune_low_magnitude_dense/AssignVariableOp*
_output_shapes
:	?*
dtype021
/prune_low_magnitude_dense/MatMul/ReadVariableOp?
 prune_low_magnitude_dense/MatMulMatMul,prune_low_magnitude_flatten/Reshape:output:07prune_low_magnitude_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 prune_low_magnitude_dense/MatMul?
0prune_low_magnitude_dense/BiasAdd/ReadVariableOpReadVariableOp9prune_low_magnitude_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0prune_low_magnitude_dense/BiasAdd/ReadVariableOp?
!prune_low_magnitude_dense/BiasAddBiasAdd*prune_low_magnitude_dense/MatMul:product:08prune_low_magnitude_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!prune_low_magnitude_dense/BiasAdd?
IdentityIdentity*prune_low_magnitude_dense/BiasAdd:output:0%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp,^prune_low_magnitude_conv2d/AssignVariableOp2^prune_low_magnitude_conv2d/BiasAdd/ReadVariableOp1^prune_low_magnitude_conv2d/Conv2D/ReadVariableOp.^prune_low_magnitude_conv2d/Mul/ReadVariableOp0^prune_low_magnitude_conv2d/Mul/ReadVariableOp_1+^prune_low_magnitude_dense/AssignVariableOp1^prune_low_magnitude_dense/BiasAdd/ReadVariableOp0^prune_low_magnitude_dense/MatMul/ReadVariableOp-^prune_low_magnitude_dense/Mul/ReadVariableOp/^prune_low_magnitude_dense/Mul/ReadVariableOp_1*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????(1: : : : : : : : 2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp2Z
+prune_low_magnitude_conv2d/AssignVariableOp+prune_low_magnitude_conv2d/AssignVariableOp2f
1prune_low_magnitude_conv2d/BiasAdd/ReadVariableOp1prune_low_magnitude_conv2d/BiasAdd/ReadVariableOp2d
0prune_low_magnitude_conv2d/Conv2D/ReadVariableOp0prune_low_magnitude_conv2d/Conv2D/ReadVariableOp2^
-prune_low_magnitude_conv2d/Mul/ReadVariableOp-prune_low_magnitude_conv2d/Mul/ReadVariableOp2b
/prune_low_magnitude_conv2d/Mul/ReadVariableOp_1/prune_low_magnitude_conv2d/Mul/ReadVariableOp_12X
*prune_low_magnitude_dense/AssignVariableOp*prune_low_magnitude_dense/AssignVariableOp2d
0prune_low_magnitude_dense/BiasAdd/ReadVariableOp0prune_low_magnitude_dense/BiasAdd/ReadVariableOp2b
/prune_low_magnitude_dense/MatMul/ReadVariableOp/prune_low_magnitude_dense/MatMul/ReadVariableOp2\
,prune_low_magnitude_dense/Mul/ReadVariableOp,prune_low_magnitude_dense/Mul/ReadVariableOp2`
.prune_low_magnitude_dense/Mul/ReadVariableOp_1.prune_low_magnitude_dense/Mul/ReadVariableOp_1:W S
/
_output_shapes
:?????????(1
 
_user_specified_nameinputs
?
?
-prune_low_magnitude_flatten_cond_false_247320V
Rprune_low_magnitude_flatten_cond_identity_prune_low_magnitude_flatten_logicaland_1
/
+prune_low_magnitude_flatten_cond_identity_1
l
%prune_low_magnitude_flatten/cond/NoOpNoOp*
_output_shapes
 2'
%prune_low_magnitude_flatten/cond/NoOp?
)prune_low_magnitude_flatten/cond/IdentityIdentityRprune_low_magnitude_flatten_cond_identity_prune_low_magnitude_flatten_logicaland_1&^prune_low_magnitude_flatten/cond/NoOp*
T0
*
_output_shapes
: 2+
)prune_low_magnitude_flatten/cond/Identity?
+prune_low_magnitude_flatten/cond/Identity_1Identity2prune_low_magnitude_flatten/cond/Identity:output:0*
T0
*
_output_shapes
: 2-
+prune_low_magnitude_flatten/cond/Identity_1"c
+prune_low_magnitude_flatten_cond_identity_14prune_low_magnitude_flatten/cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
<__inference_prune_low_magnitude_flatten_layer_call_fn_247800

inputs
unknown:	 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *`
f[RY
W__inference_prune_low_magnitude_flatten_layer_call_and_return_conditional_losses_2464462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????
: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?
?
cond_false_246628
cond_placeholder
cond_placeholder_1
cond_placeholder_2
cond_placeholder_3
cond_identity_logicaland_1

cond_identity_1
4
	cond/NoOpNoOp*
_output_shapes
 2
	cond/NoOps
cond/IdentityIdentitycond_identity_logicaland_1
^cond/NoOp*
T0
*
_output_shapes
: 2
cond/Identityg
cond/Identity_1Identitycond/Identity:output:0*
T0
*
_output_shapes
: 2
cond/Identity_1"+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : :

_output_shapes
: 
Ԉ
?
"__inference__traced_restore_248270
file_prefix#
assignvariableop_mean:)
assignvariableop_1_variance:"
assignvariableop_2_count:	 L
2assignvariableop_3_prune_low_magnitude_conv2d_mask:
A
7assignvariableop_4_prune_low_magnitude_conv2d_threshold: D
:assignvariableop_5_prune_low_magnitude_conv2d_pruning_step:	 K
Aassignvariableop_6_prune_low_magnitude_max_pooling2d_pruning_step:	 E
;assignvariableop_7_prune_low_magnitude_flatten_pruning_step:	 D
1assignvariableop_8_prune_low_magnitude_dense_mask:	?@
6assignvariableop_9_prune_low_magnitude_dense_threshold: D
:assignvariableop_10_prune_low_magnitude_dense_pruning_step:	 "
assignvariableop_11_iter:	 $
assignvariableop_12_beta_1: $
assignvariableop_13_beta_2: #
assignvariableop_14_decay: +
!assignvariableop_15_learning_rate: ;
!assignvariableop_16_conv2d_kernel:
-
assignvariableop_17_conv2d_bias:3
 assignvariableop_18_dense_kernel:	?,
assignvariableop_19_dense_bias:#
assignvariableop_20_total: %
assignvariableop_21_count_1: %
assignvariableop_22_total_1: %
assignvariableop_23_count_2: B
(assignvariableop_24_adam_conv2d_kernel_m:
4
&assignvariableop_25_adam_conv2d_bias_m::
'assignvariableop_26_adam_dense_kernel_m:	?3
%assignvariableop_27_adam_dense_bias_m:B
(assignvariableop_28_adam_conv2d_kernel_v:
4
&assignvariableop_29_adam_conv2d_bias_v::
'assignvariableop_30_adam_dense_kernel_v:	?3
%assignvariableop_31_adam_dense_bias_v:
identity_33??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*?
value?B?!B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/mask/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-1/threshold/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-1/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-2/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-3/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/mask/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-4/threshold/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-4/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?:::::::::::::::::::::::::::::::::*/
dtypes%
#2!						2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_meanIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_varianceIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_countIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp2assignvariableop_3_prune_low_magnitude_conv2d_maskIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp7assignvariableop_4_prune_low_magnitude_conv2d_thresholdIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp:assignvariableop_5_prune_low_magnitude_conv2d_pruning_stepIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOpAassignvariableop_6_prune_low_magnitude_max_pooling2d_pruning_stepIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0	*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp;assignvariableop_7_prune_low_magnitude_flatten_pruning_stepIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp1assignvariableop_8_prune_low_magnitude_dense_maskIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp6assignvariableop_9_prune_low_magnitude_dense_thresholdIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp:assignvariableop_10_prune_low_magnitude_dense_pruning_stepIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_iterIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_beta_1Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_beta_2Identity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOpassignvariableop_14_decayIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp!assignvariableop_15_learning_rateIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp!assignvariableop_16_conv2d_kernelIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_conv2d_biasIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp assignvariableop_18_dense_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_dense_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_totalIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_count_1Identity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_total_1Identity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpassignvariableop_23_count_2Identity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp(assignvariableop_24_adam_conv2d_kernel_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp&assignvariableop_25_adam_conv2d_bias_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp'assignvariableop_26_adam_dense_kernel_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp%assignvariableop_27_adam_dense_bias_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp(assignvariableop_28_adam_conv2d_kernel_vIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp&assignvariableop_29_adam_conv2d_bias_vIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp'assignvariableop_30_adam_dense_kernel_vIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp%assignvariableop_31_adam_dense_bias_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_319
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_32Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_32?
Identity_33IdentityIdentity_32:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_33"#
identity_33Identity_33:output:0*U
_input_shapesD
B: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
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
AssignVariableOp_31AssignVariableOp_312(
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
?	
?
3assert_greater_equal_Assert_AssertGuard_true_247561M
Iassert_greater_equal_assert_assertguard_identity_assert_greater_equal_all
7
3assert_greater_equal_assert_assertguard_placeholder	9
5assert_greater_equal_assert_assertguard_placeholder_1	6
2assert_greater_equal_assert_assertguard_identity_1
z
,assert_greater_equal/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2.
,assert_greater_equal/Assert/AssertGuard/NoOp?
0assert_greater_equal/Assert/AssertGuard/IdentityIdentityIassert_greater_equal_assert_assertguard_identity_assert_greater_equal_all-^assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identity?
2assert_greater_equal/Assert/AssertGuard/Identity_1Identity9assert_greater_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 24
2assert_greater_equal/Assert/AssertGuard/Identity_1"q
2assert_greater_equal_assert_assertguard_identity_1;assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?G
?
U__inference_prune_low_magnitude_dense_layer_call_and_return_conditional_losses_248045

inputs6
,assert_greater_equal_readvariableop_resource:	 
cond_input_1:	?
cond_input_2:	?
cond_input_3: -
biasadd_readvariableop_resource:
identity??AssignVariableOp?BiasAdd/ReadVariableOp?GreaterEqual/ReadVariableOp?LessEqual/ReadVariableOp?MatMul/ReadVariableOp?Mul/ReadVariableOp?Mul/ReadVariableOp_1?Sub/ReadVariableOp?'assert_greater_equal/Assert/AssertGuard?#assert_greater_equal/ReadVariableOp?cond?
#assert_greater_equal/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource*
_output_shapes
: *
dtype0	2%
#assert_greater_equal/ReadVariableOpr
assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
assert_greater_equal/y?
!assert_greater_equal/GreaterEqualGreaterEqual+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
T0	*
_output_shapes
: 2#
!assert_greater_equal/GreaterEqualx
assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 2
assert_greater_equal/Rank?
 assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2"
 assert_greater_equal/range/start?
 assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2"
 assert_greater_equal/range/delta?
assert_greater_equal/rangeRange)assert_greater_equal/range/start:output:0"assert_greater_equal/Rank:output:0)assert_greater_equal/range/delta:output:0*
_output_shapes
: 2
assert_greater_equal/range?
assert_greater_equal/AllAll%assert_greater_equal/GreaterEqual:z:0#assert_greater_equal/range:output:0*
_output_shapes
: 2
assert_greater_equal/All?
!assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*?
value?B? B?Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2#
!assert_greater_equal/Assert/Const?
#assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2%
#assert_greater_equal/Assert/Const_1?
#assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2%
#assert_greater_equal/Assert/Const_2?
#assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2%
#assert_greater_equal/Assert/Const_3?
'assert_greater_equal/Assert/AssertGuardIf!assert_greater_equal/All:output:0!assert_greater_equal/All:output:0+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *G
else_branch8R6
4assert_greater_equal_Assert_AssertGuard_false_247934*
output_shapes
: *F
then_branch7R5
3assert_greater_equal_Assert_AssertGuard_true_2479332)
'assert_greater_equal/Assert/AssertGuard?
0assert_greater_equal/Assert/AssertGuard/IdentityIdentity0assert_greater_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identity?
GreaterEqual/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOp?
GreaterEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y?
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqual?
LessEqual/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp?
LessEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqual?
Less/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB :
?????????2
Less/x?
Less/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd?
Sub/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOp?
Sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
Sub?

FloorMod/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorMod?
Equal/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1?
ConstConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Const?
condIfLogicalAnd_1:z:0,assert_greater_equal_readvariableop_resourcecond_input_1cond_input_2cond_input_3LogicalAnd_1:z:0*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: *$
_read_only_resource_inputs
*$
else_branchR
cond_false_247974*
output_shapes
: *#
then_branchR
cond_true_2479732
condZ
cond/IdentityIdentitycond:output:0*
T0
*
_output_shapes
: 2
cond/Identityq
updateNoOp1^assert_greater_equal/Assert/AssertGuard/Identity^cond/Identity*
_output_shapes
 2
updatev
Mul/ReadVariableOpReadVariableOpcond_input_1*
_output_shapes
:	?*
dtype02
Mul/ReadVariableOp?
Mul/ReadVariableOp_1ReadVariableOpcond_input_2^cond*
_output_shapes
:	?*
dtype02
Mul/ReadVariableOp_1u
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	?2
Mul?
AssignVariableOpAssignVariableOpcond_input_1Mul:z:0^Mul/ReadVariableOp^cond*
_output_shapes
 *
dtype02
AssignVariableOpw

group_depsNoOp^AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2

group_depsu
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2
group_deps_1?
MatMul/ReadVariableOpReadVariableOpcond_input_1^AssignVariableOp*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
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
BiasAdd?
IdentityIdentityBiasAdd:output:0^AssignVariableOp^BiasAdd/ReadVariableOp^GreaterEqual/ReadVariableOp^LessEqual/ReadVariableOp^MatMul/ReadVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1^Sub/ReadVariableOp(^assert_greater_equal/Assert/AssertGuard$^assert_greater_equal/ReadVariableOp^cond*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : : : 2$
AssignVariableOpAssignVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2:
GreaterEqual/ReadVariableOpGreaterEqual/ReadVariableOp24
LessEqual/ReadVariableOpLessEqual/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_12(
Sub/ReadVariableOpSub/ReadVariableOp2R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard2J
#assert_greater_equal/ReadVariableOp#assert_greater_equal/ReadVariableOp2
condcond:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
X
<__inference_prune_low_magnitude_flatten_layer_call_fn_247793

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
GPU 2J 8? *`
f[RY
W__inference_prune_low_magnitude_flatten_layer_call_and_return_conditional_losses_2461682
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
?
G
cond_true_246431
cond_identity_logicaland_1

cond_identity_1
@
cond/group_depsNoOp*
_output_shapes
 2
cond/group_depsy
cond/IdentityIdentitycond_identity_logicaland_1^cond/group_deps*
T0
*
_output_shapes
: 2
cond/Identityg
cond/Identity_1Identitycond/Identity:output:0*
T0
*
_output_shapes
: 2
cond/Identity_1"+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?9
?
W__inference_prune_low_magnitude_flatten_layer_call_and_return_conditional_losses_246446

inputs6
,assert_greater_equal_readvariableop_resource:	 
identity??GreaterEqual/ReadVariableOp?LessEqual/ReadVariableOp?Sub/ReadVariableOp?'assert_greater_equal/Assert/AssertGuard?#assert_greater_equal/ReadVariableOp?
#assert_greater_equal/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource*
_output_shapes
: *
dtype0	2%
#assert_greater_equal/ReadVariableOpr
assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
assert_greater_equal/y?
!assert_greater_equal/GreaterEqualGreaterEqual+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
T0	*
_output_shapes
: 2#
!assert_greater_equal/GreaterEqualx
assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 2
assert_greater_equal/Rank?
 assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2"
 assert_greater_equal/range/start?
 assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2"
 assert_greater_equal/range/delta?
assert_greater_equal/rangeRange)assert_greater_equal/range/start:output:0"assert_greater_equal/Rank:output:0)assert_greater_equal/range/delta:output:0*
_output_shapes
: 2
assert_greater_equal/range?
assert_greater_equal/AllAll%assert_greater_equal/GreaterEqual:z:0#assert_greater_equal/range:output:0*
_output_shapes
: 2
assert_greater_equal/All?
!assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*?
value?B? B?Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2#
!assert_greater_equal/Assert/Const?
#assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2%
#assert_greater_equal/Assert/Const_1?
#assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2%
#assert_greater_equal/Assert/Const_2?
#assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2%
#assert_greater_equal/Assert/Const_3?
'assert_greater_equal/Assert/AssertGuardIf!assert_greater_equal/All:output:0!assert_greater_equal/All:output:0+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *G
else_branch8R6
4assert_greater_equal_Assert_AssertGuard_false_246392*
output_shapes
: *F
then_branch7R5
3assert_greater_equal_Assert_AssertGuard_true_2463912)
'assert_greater_equal/Assert/AssertGuard?
0assert_greater_equal/Assert/AssertGuard/IdentityIdentity0assert_greater_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identity?
GreaterEqual/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOp?
GreaterEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y?
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqual?
LessEqual/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp?
LessEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqual?
Less/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB :
?????????2
Less/x?
Less/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd?
Sub/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOp?
Sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
Sub?

FloorMod/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorMod?
Equal/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1?
ConstConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Const?
condStatelessIfLogicalAnd_1:z:0LogicalAnd_1:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *$
else_branchR
cond_false_246432*
output_shapes
: *#
then_branchR
cond_true_2464312
condZ
cond/IdentityIdentitycond:output:0*
T0
*
_output_shapes
: 2
cond/Identityq
updateNoOp1^assert_greater_equal/Assert/AssertGuard/Identity^cond/Identity*
_output_shapes
 2
update6

group_depsNoOp*
_output_shapes
 2

group_depsc
Const_1Const*
_output_shapes
:*
dtype0*
valueB"?????  2	
Const_1j
ReshapeReshapeinputsConst_1:output:0*
T0*(
_output_shapes
:??????????2	
Reshape?
IdentityIdentityReshape:output:0^GreaterEqual/ReadVariableOp^LessEqual/ReadVariableOp^Sub/ReadVariableOp(^assert_greater_equal/Assert/AssertGuard$^assert_greater_equal/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????
: 2:
GreaterEqual/ReadVariableOpGreaterEqual/ReadVariableOp24
LessEqual/ReadVariableOpLessEqual/ReadVariableOp2(
Sub/ReadVariableOpSub/ReadVariableOp2R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard2J
#assert_greater_equal/ReadVariableOp#assert_greater_equal/ReadVariableOp:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?9
?
]__inference_prune_low_magnitude_max_pooling2d_layer_call_and_return_conditional_losses_246552

inputs6
,assert_greater_equal_readvariableop_resource:	 
identity??GreaterEqual/ReadVariableOp?LessEqual/ReadVariableOp?Sub/ReadVariableOp?'assert_greater_equal/Assert/AssertGuard?#assert_greater_equal/ReadVariableOp?
#assert_greater_equal/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource*
_output_shapes
: *
dtype0	2%
#assert_greater_equal/ReadVariableOpr
assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
assert_greater_equal/y?
!assert_greater_equal/GreaterEqualGreaterEqual+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
T0	*
_output_shapes
: 2#
!assert_greater_equal/GreaterEqualx
assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 2
assert_greater_equal/Rank?
 assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2"
 assert_greater_equal/range/start?
 assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2"
 assert_greater_equal/range/delta?
assert_greater_equal/rangeRange)assert_greater_equal/range/start:output:0"assert_greater_equal/Rank:output:0)assert_greater_equal/range/delta:output:0*
_output_shapes
: 2
assert_greater_equal/range?
assert_greater_equal/AllAll%assert_greater_equal/GreaterEqual:z:0#assert_greater_equal/range:output:0*
_output_shapes
: 2
assert_greater_equal/All?
!assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*?
value?B? B?Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2#
!assert_greater_equal/Assert/Const?
#assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2%
#assert_greater_equal/Assert/Const_1?
#assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2%
#assert_greater_equal/Assert/Const_2?
#assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2%
#assert_greater_equal/Assert/Const_3?
'assert_greater_equal/Assert/AssertGuardIf!assert_greater_equal/All:output:0!assert_greater_equal/All:output:0+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *G
else_branch8R6
4assert_greater_equal_Assert_AssertGuard_false_246499*
output_shapes
: *F
then_branch7R5
3assert_greater_equal_Assert_AssertGuard_true_2464982)
'assert_greater_equal/Assert/AssertGuard?
0assert_greater_equal/Assert/AssertGuard/IdentityIdentity0assert_greater_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identity?
GreaterEqual/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOp?
GreaterEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y?
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqual?
LessEqual/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp?
LessEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqual?
Less/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB :
?????????2
Less/x?
Less/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd?
Sub/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOp?
Sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
Sub?

FloorMod/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorMod?
Equal/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1?
ConstConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Const?
condStatelessIfLogicalAnd_1:z:0LogicalAnd_1:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *$
else_branchR
cond_false_246539*
output_shapes
: *#
then_branchR
cond_true_2465382
condZ
cond/IdentityIdentitycond:output:0*
T0
*
_output_shapes
: 2
cond/Identityq
updateNoOp1^assert_greater_equal/Assert/AssertGuard/Identity^cond/Identity*
_output_shapes
 2
update6

group_depsNoOp*
_output_shapes
 2

group_deps?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????
*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0^GreaterEqual/ReadVariableOp^LessEqual/ReadVariableOp^Sub/ReadVariableOp(^assert_greater_equal/Assert/AssertGuard$^assert_greater_equal/ReadVariableOp*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: 2:
GreaterEqual/ReadVariableOpGreaterEqual/ReadVariableOp24
LessEqual/ReadVariableOpLessEqual/ReadVariableOp2(
Sub/ReadVariableOpSub/ReadVariableOp2R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard2J
#assert_greater_equal/ReadVariableOp#assert_greater_equal/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
Nprune_low_magnitude_conv2d_assert_greater_equal_Assert_AssertGuard_true_247080?
prune_low_magnitude_conv2d_assert_greater_equal_assert_assertguard_identity_prune_low_magnitude_conv2d_assert_greater_equal_all
R
Nprune_low_magnitude_conv2d_assert_greater_equal_assert_assertguard_placeholder	T
Pprune_low_magnitude_conv2d_assert_greater_equal_assert_assertguard_placeholder_1	Q
Mprune_low_magnitude_conv2d_assert_greater_equal_assert_assertguard_identity_1
?
Gprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2I
Gprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/NoOp?
Kprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/IdentityIdentityprune_low_magnitude_conv2d_assert_greater_equal_assert_assertguard_identity_prune_low_magnitude_conv2d_assert_greater_equal_allH^prune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 2M
Kprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Identity?
Mprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Identity_1IdentityTprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 2O
Mprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Identity_1"?
Mprune_low_magnitude_conv2d_assert_greater_equal_assert_assertguard_identity_1Vprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?9
?
W__inference_prune_low_magnitude_flatten_layer_call_and_return_conditional_losses_247877

inputs6
,assert_greater_equal_readvariableop_resource:	 
identity??GreaterEqual/ReadVariableOp?LessEqual/ReadVariableOp?Sub/ReadVariableOp?'assert_greater_equal/Assert/AssertGuard?#assert_greater_equal/ReadVariableOp?
#assert_greater_equal/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource*
_output_shapes
: *
dtype0	2%
#assert_greater_equal/ReadVariableOpr
assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
assert_greater_equal/y?
!assert_greater_equal/GreaterEqualGreaterEqual+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
T0	*
_output_shapes
: 2#
!assert_greater_equal/GreaterEqualx
assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 2
assert_greater_equal/Rank?
 assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2"
 assert_greater_equal/range/start?
 assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2"
 assert_greater_equal/range/delta?
assert_greater_equal/rangeRange)assert_greater_equal/range/start:output:0"assert_greater_equal/Rank:output:0)assert_greater_equal/range/delta:output:0*
_output_shapes
: 2
assert_greater_equal/range?
assert_greater_equal/AllAll%assert_greater_equal/GreaterEqual:z:0#assert_greater_equal/range:output:0*
_output_shapes
: 2
assert_greater_equal/All?
!assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*?
value?B? B?Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2#
!assert_greater_equal/Assert/Const?
#assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2%
#assert_greater_equal/Assert/Const_1?
#assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2%
#assert_greater_equal/Assert/Const_2?
#assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2%
#assert_greater_equal/Assert/Const_3?
'assert_greater_equal/Assert/AssertGuardIf!assert_greater_equal/All:output:0!assert_greater_equal/All:output:0+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *G
else_branch8R6
4assert_greater_equal_Assert_AssertGuard_false_247823*
output_shapes
: *F
then_branch7R5
3assert_greater_equal_Assert_AssertGuard_true_2478222)
'assert_greater_equal/Assert/AssertGuard?
0assert_greater_equal/Assert/AssertGuard/IdentityIdentity0assert_greater_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identity?
GreaterEqual/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOp?
GreaterEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y?
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqual?
LessEqual/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp?
LessEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqual?
Less/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB :
?????????2
Less/x?
Less/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd?
Sub/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOp?
Sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
Sub?

FloorMod/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorMod?
Equal/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1?
ConstConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Const?
condStatelessIfLogicalAnd_1:z:0LogicalAnd_1:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *$
else_branchR
cond_false_247863*
output_shapes
: *#
then_branchR
cond_true_2478622
condZ
cond/IdentityIdentitycond:output:0*
T0
*
_output_shapes
: 2
cond/Identityq
updateNoOp1^assert_greater_equal/Assert/AssertGuard/Identity^cond/Identity*
_output_shapes
 2
update6

group_depsNoOp*
_output_shapes
 2

group_depsc
Const_1Const*
_output_shapes
:*
dtype0*
valueB"?????  2	
Const_1j
ReshapeReshapeinputsConst_1:output:0*
T0*(
_output_shapes
:??????????2	
Reshape?
IdentityIdentityReshape:output:0^GreaterEqual/ReadVariableOp^LessEqual/ReadVariableOp^Sub/ReadVariableOp(^assert_greater_equal/Assert/AssertGuard$^assert_greater_equal/ReadVariableOp*
T0*(
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????
: 2:
GreaterEqual/ReadVariableOpGreaterEqual/ReadVariableOp24
LessEqual/ReadVariableOpLessEqual/ReadVariableOp2(
Sub/ReadVariableOpSub/ReadVariableOp2R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard2J
#assert_greater_equal/ReadVariableOp#assert_greater_equal/ReadVariableOp:W S
/
_output_shapes
:?????????

 
_user_specified_nameinputs
?	
?
3assert_greater_equal_Assert_AssertGuard_true_247933M
Iassert_greater_equal_assert_assertguard_identity_assert_greater_equal_all
7
3assert_greater_equal_assert_assertguard_placeholder	9
5assert_greater_equal_assert_assertguard_placeholder_1	6
2assert_greater_equal_assert_assertguard_identity_1
z
,assert_greater_equal/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2.
,assert_greater_equal/Assert/AssertGuard/NoOp?
0assert_greater_equal/Assert/AssertGuard/IdentityIdentityIassert_greater_equal_assert_assertguard_identity_assert_greater_equal_all-^assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identity?
2assert_greater_equal/Assert/AssertGuard/Identity_1Identity9assert_greater_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 24
2assert_greater_equal/Assert/AssertGuard/Identity_1"q
2assert_greater_equal_assert_assertguard_identity_1;assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
$__inference_signature_wrapper_246955
input_1
unknown:
	unknown_0:#
	unknown_1:
#
	unknown_2:

	unknown_3:
	unknown_4:	?
	unknown_5:	?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? **
f%R#
!__inference__wrapped_model_2460932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????(1: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????(1
!
_user_specified_name	input_1
?	
?
3assert_greater_equal_Assert_AssertGuard_true_246587M
Iassert_greater_equal_assert_assertguard_identity_assert_greater_equal_all
7
3assert_greater_equal_assert_assertguard_placeholder	9
5assert_greater_equal_assert_assertguard_placeholder_1	6
2assert_greater_equal_assert_assertguard_identity_1
z
,assert_greater_equal/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2.
,assert_greater_equal/Assert/AssertGuard/NoOp?
0assert_greater_equal/Assert/AssertGuard/IdentityIdentityIassert_greater_equal_assert_assertguard_identity_assert_greater_equal_all-^assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identity?
2assert_greater_equal/Assert/AssertGuard/Identity_1Identity9assert_greater_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 24
2assert_greater_equal/Assert/AssertGuard/Identity_1"q
2assert_greater_equal_assert_assertguard_identity_1;assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
3assert_greater_equal_Assert_AssertGuard_true_246242M
Iassert_greater_equal_assert_assertguard_identity_assert_greater_equal_all
7
3assert_greater_equal_assert_assertguard_placeholder	9
5assert_greater_equal_assert_assertguard_placeholder_1	6
2assert_greater_equal_assert_assertguard_identity_1
z
,assert_greater_equal/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2.
,assert_greater_equal/Assert/AssertGuard/NoOp?
0assert_greater_equal/Assert/AssertGuard/IdentityIdentityIassert_greater_equal_assert_assertguard_identity_assert_greater_equal_all-^assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identity?
2assert_greater_equal/Assert/AssertGuard/Identity_1Identity9assert_greater_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 24
2assert_greater_equal/Assert/AssertGuard/Identity_1"q
2assert_greater_equal_assert_assertguard_identity_1;assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
? 
?
Oprune_low_magnitude_conv2d_assert_greater_equal_Assert_AssertGuard_false_247081?
}prune_low_magnitude_conv2d_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_conv2d_assert_greater_equal_all
?
?prune_low_magnitude_conv2d_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_conv2d_assert_greater_equal_readvariableop	
{prune_low_magnitude_conv2d_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_conv2d_assert_greater_equal_y	Q
Mprune_low_magnitude_conv2d_assert_greater_equal_assert_assertguard_identity_1
??Iprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Assert?
Pprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*?
value?B? B?Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2R
Pprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Assert/data_0?
Pprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2R
Pprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Assert/data_1?
Pprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*X
valueOBM BGx (prune_low_magnitude_conv2d/assert_greater_equal/ReadVariableOp:0) = 2R
Pprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Assert/data_2?
Pprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*K
valueBB@ B:y (prune_low_magnitude_conv2d/assert_greater_equal/y:0) = 2R
Pprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Assert/data_4?
Iprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/AssertAssert}prune_low_magnitude_conv2d_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_conv2d_assert_greater_equal_allYprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Assert/data_0:output:0Yprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Assert/data_1:output:0Yprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Assert/data_2:output:0?prune_low_magnitude_conv2d_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_conv2d_assert_greater_equal_readvariableopYprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Assert/data_4:output:0{prune_low_magnitude_conv2d_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_conv2d_assert_greater_equal_y*
T

2		*
_output_shapes
 2K
Iprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Assert?
Kprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/IdentityIdentity}prune_low_magnitude_conv2d_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_conv2d_assert_greater_equal_allJ^prune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2M
Kprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Identity?
Mprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Identity_1IdentityTprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Identity:output:0J^prune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2O
Mprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Identity_1"?
Mprune_low_magnitude_conv2d_assert_greater_equal_assert_assertguard_identity_1Vprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2?
Iprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/AssertIprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
??
?
Q__inference_ExtAudioDataModelPrun_layer_call_and_return_conditional_losses_247458

inputs;
-normalization_reshape_readvariableop_resource:=
/normalization_reshape_1_readvariableop_resource:Q
Gprune_low_magnitude_conv2d_assert_greater_equal_readvariableop_resource:	 A
'prune_low_magnitude_conv2d_cond_input_1:
A
'prune_low_magnitude_conv2d_cond_input_2:
1
'prune_low_magnitude_conv2d_cond_input_3: H
:prune_low_magnitude_conv2d_biasadd_readvariableop_resource:X
Nprune_low_magnitude_max_pooling2d_assert_greater_equal_readvariableop_resource:	 R
Hprune_low_magnitude_flatten_assert_greater_equal_readvariableop_resource:	 P
Fprune_low_magnitude_dense_assert_greater_equal_readvariableop_resource:	 9
&prune_low_magnitude_dense_cond_input_1:	?9
&prune_low_magnitude_dense_cond_input_2:	?0
&prune_low_magnitude_dense_cond_input_3: G
9prune_low_magnitude_dense_biasadd_readvariableop_resource:
identity??$normalization/Reshape/ReadVariableOp?&normalization/Reshape_1/ReadVariableOp?+prune_low_magnitude_conv2d/AssignVariableOp?1prune_low_magnitude_conv2d/BiasAdd/ReadVariableOp?0prune_low_magnitude_conv2d/Conv2D/ReadVariableOp?6prune_low_magnitude_conv2d/GreaterEqual/ReadVariableOp?3prune_low_magnitude_conv2d/LessEqual/ReadVariableOp?-prune_low_magnitude_conv2d/Mul/ReadVariableOp?/prune_low_magnitude_conv2d/Mul/ReadVariableOp_1?-prune_low_magnitude_conv2d/Sub/ReadVariableOp?Bprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard?>prune_low_magnitude_conv2d/assert_greater_equal/ReadVariableOp?prune_low_magnitude_conv2d/cond?*prune_low_magnitude_dense/AssignVariableOp?0prune_low_magnitude_dense/BiasAdd/ReadVariableOp?5prune_low_magnitude_dense/GreaterEqual/ReadVariableOp?2prune_low_magnitude_dense/LessEqual/ReadVariableOp?/prune_low_magnitude_dense/MatMul/ReadVariableOp?,prune_low_magnitude_dense/Mul/ReadVariableOp?.prune_low_magnitude_dense/Mul/ReadVariableOp_1?,prune_low_magnitude_dense/Sub/ReadVariableOp?Aprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard?=prune_low_magnitude_dense/assert_greater_equal/ReadVariableOp?prune_low_magnitude_dense/cond?7prune_low_magnitude_flatten/GreaterEqual/ReadVariableOp?4prune_low_magnitude_flatten/LessEqual/ReadVariableOp?.prune_low_magnitude_flatten/Sub/ReadVariableOp?Cprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard??prune_low_magnitude_flatten/assert_greater_equal/ReadVariableOp?=prune_low_magnitude_max_pooling2d/GreaterEqual/ReadVariableOp?:prune_low_magnitude_max_pooling2d/LessEqual/ReadVariableOp?4prune_low_magnitude_max_pooling2d/Sub/ReadVariableOp?Iprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard?Eprune_low_magnitude_max_pooling2d/assert_greater_equal/ReadVariableOp?
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp?
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape/shape?
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:2
normalization/Reshape?
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp?
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape_1/shape?
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
normalization/Reshape_1?
normalization/subSubinputsnormalization/Reshape:output:0*
T0*/
_output_shapes
:?????????(12
normalization/sub?
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*&
_output_shapes
:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*&
_output_shapes
:2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*/
_output_shapes
:?????????(12
normalization/truediv?
>prune_low_magnitude_conv2d/assert_greater_equal/ReadVariableOpReadVariableOpGprune_low_magnitude_conv2d_assert_greater_equal_readvariableop_resource*
_output_shapes
: *
dtype0	2@
>prune_low_magnitude_conv2d/assert_greater_equal/ReadVariableOp?
1prune_low_magnitude_conv2d/assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 23
1prune_low_magnitude_conv2d/assert_greater_equal/y?
<prune_low_magnitude_conv2d/assert_greater_equal/GreaterEqualGreaterEqualFprune_low_magnitude_conv2d/assert_greater_equal/ReadVariableOp:value:0:prune_low_magnitude_conv2d/assert_greater_equal/y:output:0*
T0	*
_output_shapes
: 2>
<prune_low_magnitude_conv2d/assert_greater_equal/GreaterEqual?
4prune_low_magnitude_conv2d/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 26
4prune_low_magnitude_conv2d/assert_greater_equal/Rank?
;prune_low_magnitude_conv2d/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2=
;prune_low_magnitude_conv2d/assert_greater_equal/range/start?
;prune_low_magnitude_conv2d/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2=
;prune_low_magnitude_conv2d/assert_greater_equal/range/delta?
5prune_low_magnitude_conv2d/assert_greater_equal/rangeRangeDprune_low_magnitude_conv2d/assert_greater_equal/range/start:output:0=prune_low_magnitude_conv2d/assert_greater_equal/Rank:output:0Dprune_low_magnitude_conv2d/assert_greater_equal/range/delta:output:0*
_output_shapes
: 27
5prune_low_magnitude_conv2d/assert_greater_equal/range?
3prune_low_magnitude_conv2d/assert_greater_equal/AllAll@prune_low_magnitude_conv2d/assert_greater_equal/GreaterEqual:z:0>prune_low_magnitude_conv2d/assert_greater_equal/range:output:0*
_output_shapes
: 25
3prune_low_magnitude_conv2d/assert_greater_equal/All?
<prune_low_magnitude_conv2d/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*?
value?B? B?Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2>
<prune_low_magnitude_conv2d/assert_greater_equal/Assert/Const?
>prune_low_magnitude_conv2d/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2@
>prune_low_magnitude_conv2d/assert_greater_equal/Assert/Const_1?
>prune_low_magnitude_conv2d/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*X
valueOBM BGx (prune_low_magnitude_conv2d/assert_greater_equal/ReadVariableOp:0) = 2@
>prune_low_magnitude_conv2d/assert_greater_equal/Assert/Const_2?
>prune_low_magnitude_conv2d/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*K
valueBB@ B:y (prune_low_magnitude_conv2d/assert_greater_equal/y:0) = 2@
>prune_low_magnitude_conv2d/assert_greater_equal/Assert/Const_3?
Bprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuardIf<prune_low_magnitude_conv2d/assert_greater_equal/All:output:0<prune_low_magnitude_conv2d/assert_greater_equal/All:output:0Fprune_low_magnitude_conv2d/assert_greater_equal/ReadVariableOp:value:0:prune_low_magnitude_conv2d/assert_greater_equal/y:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *b
else_branchSRQ
Oprune_low_magnitude_conv2d_assert_greater_equal_Assert_AssertGuard_false_247081*
output_shapes
: *a
then_branchRRP
Nprune_low_magnitude_conv2d_assert_greater_equal_Assert_AssertGuard_true_2470802D
Bprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard?
Kprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/IdentityIdentityKprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 2M
Kprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Identity?
6prune_low_magnitude_conv2d/GreaterEqual/ReadVariableOpReadVariableOpGprune_low_magnitude_conv2d_assert_greater_equal_readvariableop_resourceL^prune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	28
6prune_low_magnitude_conv2d/GreaterEqual/ReadVariableOp?
)prune_low_magnitude_conv2d/GreaterEqual/yConstL^prune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2+
)prune_low_magnitude_conv2d/GreaterEqual/y?
'prune_low_magnitude_conv2d/GreaterEqualGreaterEqual>prune_low_magnitude_conv2d/GreaterEqual/ReadVariableOp:value:02prune_low_magnitude_conv2d/GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2)
'prune_low_magnitude_conv2d/GreaterEqual?
3prune_low_magnitude_conv2d/LessEqual/ReadVariableOpReadVariableOpGprune_low_magnitude_conv2d_assert_greater_equal_readvariableop_resourceL^prune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	25
3prune_low_magnitude_conv2d/LessEqual/ReadVariableOp?
&prune_low_magnitude_conv2d/LessEqual/yConstL^prune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2(
&prune_low_magnitude_conv2d/LessEqual/y?
$prune_low_magnitude_conv2d/LessEqual	LessEqual;prune_low_magnitude_conv2d/LessEqual/ReadVariableOp:value:0/prune_low_magnitude_conv2d/LessEqual/y:output:0*
T0	*
_output_shapes
: 2&
$prune_low_magnitude_conv2d/LessEqual?
!prune_low_magnitude_conv2d/Less/xConstL^prune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB :
?????????2#
!prune_low_magnitude_conv2d/Less/x?
!prune_low_magnitude_conv2d/Less/yConstL^prune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : 2#
!prune_low_magnitude_conv2d/Less/y?
prune_low_magnitude_conv2d/LessLess*prune_low_magnitude_conv2d/Less/x:output:0*prune_low_magnitude_conv2d/Less/y:output:0*
T0*
_output_shapes
: 2!
prune_low_magnitude_conv2d/Less?
$prune_low_magnitude_conv2d/LogicalOr	LogicalOr(prune_low_magnitude_conv2d/LessEqual:z:0#prune_low_magnitude_conv2d/Less:z:0*
_output_shapes
: 2&
$prune_low_magnitude_conv2d/LogicalOr?
%prune_low_magnitude_conv2d/LogicalAnd
LogicalAnd+prune_low_magnitude_conv2d/GreaterEqual:z:0(prune_low_magnitude_conv2d/LogicalOr:z:0*
_output_shapes
: 2'
%prune_low_magnitude_conv2d/LogicalAnd?
-prune_low_magnitude_conv2d/Sub/ReadVariableOpReadVariableOpGprune_low_magnitude_conv2d_assert_greater_equal_readvariableop_resourceL^prune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2/
-prune_low_magnitude_conv2d/Sub/ReadVariableOp?
 prune_low_magnitude_conv2d/Sub/yConstL^prune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2"
 prune_low_magnitude_conv2d/Sub/y?
prune_low_magnitude_conv2d/SubSub5prune_low_magnitude_conv2d/Sub/ReadVariableOp:value:0)prune_low_magnitude_conv2d/Sub/y:output:0*
T0	*
_output_shapes
: 2 
prune_low_magnitude_conv2d/Sub?
%prune_low_magnitude_conv2d/FloorMod/yConstL^prune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd2'
%prune_low_magnitude_conv2d/FloorMod/y?
#prune_low_magnitude_conv2d/FloorModFloorMod"prune_low_magnitude_conv2d/Sub:z:0.prune_low_magnitude_conv2d/FloorMod/y:output:0*
T0	*
_output_shapes
: 2%
#prune_low_magnitude_conv2d/FloorMod?
"prune_low_magnitude_conv2d/Equal/yConstL^prune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2$
"prune_low_magnitude_conv2d/Equal/y?
 prune_low_magnitude_conv2d/EqualEqual'prune_low_magnitude_conv2d/FloorMod:z:0+prune_low_magnitude_conv2d/Equal/y:output:0*
T0	*
_output_shapes
: 2"
 prune_low_magnitude_conv2d/Equal?
'prune_low_magnitude_conv2d/LogicalAnd_1
LogicalAnd)prune_low_magnitude_conv2d/LogicalAnd:z:0$prune_low_magnitude_conv2d/Equal:z:0*
_output_shapes
: 2)
'prune_low_magnitude_conv2d/LogicalAnd_1?
 prune_low_magnitude_conv2d/ConstConstL^prune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *   ?2"
 prune_low_magnitude_conv2d/Const?
prune_low_magnitude_conv2d/condIf+prune_low_magnitude_conv2d/LogicalAnd_1:z:0Gprune_low_magnitude_conv2d_assert_greater_equal_readvariableop_resource'prune_low_magnitude_conv2d_cond_input_1'prune_low_magnitude_conv2d_cond_input_2'prune_low_magnitude_conv2d_cond_input_3+prune_low_magnitude_conv2d/LogicalAnd_1:z:0*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: *$
_read_only_resource_inputs
*?
else_branch0R.
,prune_low_magnitude_conv2d_cond_false_247121*
output_shapes
: *>
then_branch/R-
+prune_low_magnitude_conv2d_cond_true_2471202!
prune_low_magnitude_conv2d/cond?
(prune_low_magnitude_conv2d/cond/IdentityIdentity(prune_low_magnitude_conv2d/cond:output:0*
T0
*
_output_shapes
: 2*
(prune_low_magnitude_conv2d/cond/Identity?
!prune_low_magnitude_conv2d/updateNoOpL^prune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard/Identity)^prune_low_magnitude_conv2d/cond/Identity*
_output_shapes
 2#
!prune_low_magnitude_conv2d/update?
-prune_low_magnitude_conv2d/Mul/ReadVariableOpReadVariableOp'prune_low_magnitude_conv2d_cond_input_1*&
_output_shapes
:
*
dtype02/
-prune_low_magnitude_conv2d/Mul/ReadVariableOp?
/prune_low_magnitude_conv2d/Mul/ReadVariableOp_1ReadVariableOp'prune_low_magnitude_conv2d_cond_input_2 ^prune_low_magnitude_conv2d/cond*&
_output_shapes
:
*
dtype021
/prune_low_magnitude_conv2d/Mul/ReadVariableOp_1?
prune_low_magnitude_conv2d/MulMul5prune_low_magnitude_conv2d/Mul/ReadVariableOp:value:07prune_low_magnitude_conv2d/Mul/ReadVariableOp_1:value:0*
T0*&
_output_shapes
:
2 
prune_low_magnitude_conv2d/Mul?
+prune_low_magnitude_conv2d/AssignVariableOpAssignVariableOp'prune_low_magnitude_conv2d_cond_input_1"prune_low_magnitude_conv2d/Mul:z:0.^prune_low_magnitude_conv2d/Mul/ReadVariableOp ^prune_low_magnitude_conv2d/cond*
_output_shapes
 *
dtype02-
+prune_low_magnitude_conv2d/AssignVariableOp?
%prune_low_magnitude_conv2d/group_depsNoOp,^prune_low_magnitude_conv2d/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2'
%prune_low_magnitude_conv2d/group_deps?
'prune_low_magnitude_conv2d/group_deps_1NoOp&^prune_low_magnitude_conv2d/group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2)
'prune_low_magnitude_conv2d/group_deps_1?
0prune_low_magnitude_conv2d/Conv2D/ReadVariableOpReadVariableOp'prune_low_magnitude_conv2d_cond_input_1,^prune_low_magnitude_conv2d/AssignVariableOp*&
_output_shapes
:
*
dtype022
0prune_low_magnitude_conv2d/Conv2D/ReadVariableOp?
!prune_low_magnitude_conv2d/Conv2DConv2Dnormalization/truediv:z:08prune_low_magnitude_conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????*
paddingVALID*
strides
2#
!prune_low_magnitude_conv2d/Conv2D?
1prune_low_magnitude_conv2d/BiasAdd/ReadVariableOpReadVariableOp:prune_low_magnitude_conv2d_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1prune_low_magnitude_conv2d/BiasAdd/ReadVariableOp?
"prune_low_magnitude_conv2d/BiasAddBiasAdd*prune_low_magnitude_conv2d/Conv2D:output:09prune_low_magnitude_conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2$
"prune_low_magnitude_conv2d/BiasAdd?
 prune_low_magnitude_conv2d/Relu6Relu6+prune_low_magnitude_conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2"
 prune_low_magnitude_conv2d/Relu6?
Eprune_low_magnitude_max_pooling2d/assert_greater_equal/ReadVariableOpReadVariableOpNprune_low_magnitude_max_pooling2d_assert_greater_equal_readvariableop_resource*
_output_shapes
: *
dtype0	2G
Eprune_low_magnitude_max_pooling2d/assert_greater_equal/ReadVariableOp?
8prune_low_magnitude_max_pooling2d/assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2:
8prune_low_magnitude_max_pooling2d/assert_greater_equal/y?
Cprune_low_magnitude_max_pooling2d/assert_greater_equal/GreaterEqualGreaterEqualMprune_low_magnitude_max_pooling2d/assert_greater_equal/ReadVariableOp:value:0Aprune_low_magnitude_max_pooling2d/assert_greater_equal/y:output:0*
T0	*
_output_shapes
: 2E
Cprune_low_magnitude_max_pooling2d/assert_greater_equal/GreaterEqual?
;prune_low_magnitude_max_pooling2d/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 2=
;prune_low_magnitude_max_pooling2d/assert_greater_equal/Rank?
Bprune_low_magnitude_max_pooling2d/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2D
Bprune_low_magnitude_max_pooling2d/assert_greater_equal/range/start?
Bprune_low_magnitude_max_pooling2d/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2D
Bprune_low_magnitude_max_pooling2d/assert_greater_equal/range/delta?
<prune_low_magnitude_max_pooling2d/assert_greater_equal/rangeRangeKprune_low_magnitude_max_pooling2d/assert_greater_equal/range/start:output:0Dprune_low_magnitude_max_pooling2d/assert_greater_equal/Rank:output:0Kprune_low_magnitude_max_pooling2d/assert_greater_equal/range/delta:output:0*
_output_shapes
: 2>
<prune_low_magnitude_max_pooling2d/assert_greater_equal/range?
:prune_low_magnitude_max_pooling2d/assert_greater_equal/AllAllGprune_low_magnitude_max_pooling2d/assert_greater_equal/GreaterEqual:z:0Eprune_low_magnitude_max_pooling2d/assert_greater_equal/range:output:0*
_output_shapes
: 2<
:prune_low_magnitude_max_pooling2d/assert_greater_equal/All?
Cprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*?
value?B? B?Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2E
Cprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/Const?
Eprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2G
Eprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/Const_1?
Eprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*_
valueVBT BNx (prune_low_magnitude_max_pooling2d/assert_greater_equal/ReadVariableOp:0) = 2G
Eprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/Const_2?
Eprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*R
valueIBG BAy (prune_low_magnitude_max_pooling2d/assert_greater_equal/y:0) = 2G
Eprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/Const_3?
Iprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuardIfCprune_low_magnitude_max_pooling2d/assert_greater_equal/All:output:0Cprune_low_magnitude_max_pooling2d/assert_greater_equal/All:output:0Mprune_low_magnitude_max_pooling2d/assert_greater_equal/ReadVariableOp:value:0Aprune_low_magnitude_max_pooling2d/assert_greater_equal/y:output:0C^prune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *i
else_branchZRX
Vprune_low_magnitude_max_pooling2d_assert_greater_equal_Assert_AssertGuard_false_247206*
output_shapes
: *h
then_branchYRW
Uprune_low_magnitude_max_pooling2d_assert_greater_equal_Assert_AssertGuard_true_2472052K
Iprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard?
Rprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/IdentityIdentityRprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 2T
Rprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Identity?
=prune_low_magnitude_max_pooling2d/GreaterEqual/ReadVariableOpReadVariableOpNprune_low_magnitude_max_pooling2d_assert_greater_equal_readvariableop_resourceS^prune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2?
=prune_low_magnitude_max_pooling2d/GreaterEqual/ReadVariableOp?
0prune_low_magnitude_max_pooling2d/GreaterEqual/yConstS^prune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 22
0prune_low_magnitude_max_pooling2d/GreaterEqual/y?
.prune_low_magnitude_max_pooling2d/GreaterEqualGreaterEqualEprune_low_magnitude_max_pooling2d/GreaterEqual/ReadVariableOp:value:09prune_low_magnitude_max_pooling2d/GreaterEqual/y:output:0*
T0	*
_output_shapes
: 20
.prune_low_magnitude_max_pooling2d/GreaterEqual?
:prune_low_magnitude_max_pooling2d/LessEqual/ReadVariableOpReadVariableOpNprune_low_magnitude_max_pooling2d_assert_greater_equal_readvariableop_resourceS^prune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2<
:prune_low_magnitude_max_pooling2d/LessEqual/ReadVariableOp?
-prune_low_magnitude_max_pooling2d/LessEqual/yConstS^prune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2/
-prune_low_magnitude_max_pooling2d/LessEqual/y?
+prune_low_magnitude_max_pooling2d/LessEqual	LessEqualBprune_low_magnitude_max_pooling2d/LessEqual/ReadVariableOp:value:06prune_low_magnitude_max_pooling2d/LessEqual/y:output:0*
T0	*
_output_shapes
: 2-
+prune_low_magnitude_max_pooling2d/LessEqual?
(prune_low_magnitude_max_pooling2d/Less/xConstS^prune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB :
?????????2*
(prune_low_magnitude_max_pooling2d/Less/x?
(prune_low_magnitude_max_pooling2d/Less/yConstS^prune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : 2*
(prune_low_magnitude_max_pooling2d/Less/y?
&prune_low_magnitude_max_pooling2d/LessLess1prune_low_magnitude_max_pooling2d/Less/x:output:01prune_low_magnitude_max_pooling2d/Less/y:output:0*
T0*
_output_shapes
: 2(
&prune_low_magnitude_max_pooling2d/Less?
+prune_low_magnitude_max_pooling2d/LogicalOr	LogicalOr/prune_low_magnitude_max_pooling2d/LessEqual:z:0*prune_low_magnitude_max_pooling2d/Less:z:0*
_output_shapes
: 2-
+prune_low_magnitude_max_pooling2d/LogicalOr?
,prune_low_magnitude_max_pooling2d/LogicalAnd
LogicalAnd2prune_low_magnitude_max_pooling2d/GreaterEqual:z:0/prune_low_magnitude_max_pooling2d/LogicalOr:z:0*
_output_shapes
: 2.
,prune_low_magnitude_max_pooling2d/LogicalAnd?
4prune_low_magnitude_max_pooling2d/Sub/ReadVariableOpReadVariableOpNprune_low_magnitude_max_pooling2d_assert_greater_equal_readvariableop_resourceS^prune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	26
4prune_low_magnitude_max_pooling2d/Sub/ReadVariableOp?
'prune_low_magnitude_max_pooling2d/Sub/yConstS^prune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2)
'prune_low_magnitude_max_pooling2d/Sub/y?
%prune_low_magnitude_max_pooling2d/SubSub<prune_low_magnitude_max_pooling2d/Sub/ReadVariableOp:value:00prune_low_magnitude_max_pooling2d/Sub/y:output:0*
T0	*
_output_shapes
: 2'
%prune_low_magnitude_max_pooling2d/Sub?
,prune_low_magnitude_max_pooling2d/FloorMod/yConstS^prune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd2.
,prune_low_magnitude_max_pooling2d/FloorMod/y?
*prune_low_magnitude_max_pooling2d/FloorModFloorMod)prune_low_magnitude_max_pooling2d/Sub:z:05prune_low_magnitude_max_pooling2d/FloorMod/y:output:0*
T0	*
_output_shapes
: 2,
*prune_low_magnitude_max_pooling2d/FloorMod?
)prune_low_magnitude_max_pooling2d/Equal/yConstS^prune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2+
)prune_low_magnitude_max_pooling2d/Equal/y?
'prune_low_magnitude_max_pooling2d/EqualEqual.prune_low_magnitude_max_pooling2d/FloorMod:z:02prune_low_magnitude_max_pooling2d/Equal/y:output:0*
T0	*
_output_shapes
: 2)
'prune_low_magnitude_max_pooling2d/Equal?
.prune_low_magnitude_max_pooling2d/LogicalAnd_1
LogicalAnd0prune_low_magnitude_max_pooling2d/LogicalAnd:z:0+prune_low_magnitude_max_pooling2d/Equal:z:0*
_output_shapes
: 20
.prune_low_magnitude_max_pooling2d/LogicalAnd_1?
'prune_low_magnitude_max_pooling2d/ConstConstS^prune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'prune_low_magnitude_max_pooling2d/Const?
&prune_low_magnitude_max_pooling2d/condStatelessIf2prune_low_magnitude_max_pooling2d/LogicalAnd_1:z:02prune_low_magnitude_max_pooling2d/LogicalAnd_1:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *F
else_branch7R5
3prune_low_magnitude_max_pooling2d_cond_false_247246*
output_shapes
: *E
then_branch6R4
2prune_low_magnitude_max_pooling2d_cond_true_2472452(
&prune_low_magnitude_max_pooling2d/cond?
/prune_low_magnitude_max_pooling2d/cond/IdentityIdentity/prune_low_magnitude_max_pooling2d/cond:output:0*
T0
*
_output_shapes
: 21
/prune_low_magnitude_max_pooling2d/cond/Identity?
(prune_low_magnitude_max_pooling2d/updateNoOpS^prune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Identity0^prune_low_magnitude_max_pooling2d/cond/Identity*
_output_shapes
 2*
(prune_low_magnitude_max_pooling2d/updatez
,prune_low_magnitude_max_pooling2d/group_depsNoOp*
_output_shapes
 2.
,prune_low_magnitude_max_pooling2d/group_deps?
)prune_low_magnitude_max_pooling2d/MaxPoolMaxPool.prune_low_magnitude_conv2d/Relu6:activations:0*/
_output_shapes
:?????????
*
ksize
*
paddingVALID*
strides
2+
)prune_low_magnitude_max_pooling2d/MaxPools
dropout/dropout/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *????2
dropout/dropout/Const?
dropout/dropout/MulMul2prune_low_magnitude_max_pooling2d/MaxPool:output:0dropout/dropout/Const:output:0*
T0*/
_output_shapes
:?????????
2
dropout/dropout/Mul?
dropout/dropout/ShapeShape2prune_low_magnitude_max_pooling2d/MaxPool:output:0*
T0*
_output_shapes
:2
dropout/dropout/Shape?
,dropout/dropout/random_uniform/RandomUniformRandomUniformdropout/dropout/Shape:output:0*
T0*/
_output_shapes
:?????????
*
dtype0*
seed??2.
,dropout/dropout/random_uniform/RandomUniform?
dropout/dropout/GreaterEqual/yConst*
_output_shapes
: *
dtype0*
valueB
 *  ?>2 
dropout/dropout/GreaterEqual/y?
dropout/dropout/GreaterEqualGreaterEqual5dropout/dropout/random_uniform/RandomUniform:output:0'dropout/dropout/GreaterEqual/y:output:0*
T0*/
_output_shapes
:?????????
2
dropout/dropout/GreaterEqual?
dropout/dropout/CastCast dropout/dropout/GreaterEqual:z:0*

DstT0*

SrcT0
*/
_output_shapes
:?????????
2
dropout/dropout/Cast?
dropout/dropout/Mul_1Muldropout/dropout/Mul:z:0dropout/dropout/Cast:y:0*
T0*/
_output_shapes
:?????????
2
dropout/dropout/Mul_1?
?prune_low_magnitude_flatten/assert_greater_equal/ReadVariableOpReadVariableOpHprune_low_magnitude_flatten_assert_greater_equal_readvariableop_resource*
_output_shapes
: *
dtype0	2A
?prune_low_magnitude_flatten/assert_greater_equal/ReadVariableOp?
2prune_low_magnitude_flatten/assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 24
2prune_low_magnitude_flatten/assert_greater_equal/y?
=prune_low_magnitude_flatten/assert_greater_equal/GreaterEqualGreaterEqualGprune_low_magnitude_flatten/assert_greater_equal/ReadVariableOp:value:0;prune_low_magnitude_flatten/assert_greater_equal/y:output:0*
T0	*
_output_shapes
: 2?
=prune_low_magnitude_flatten/assert_greater_equal/GreaterEqual?
5prune_low_magnitude_flatten/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 27
5prune_low_magnitude_flatten/assert_greater_equal/Rank?
<prune_low_magnitude_flatten/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2>
<prune_low_magnitude_flatten/assert_greater_equal/range/start?
<prune_low_magnitude_flatten/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2>
<prune_low_magnitude_flatten/assert_greater_equal/range/delta?
6prune_low_magnitude_flatten/assert_greater_equal/rangeRangeEprune_low_magnitude_flatten/assert_greater_equal/range/start:output:0>prune_low_magnitude_flatten/assert_greater_equal/Rank:output:0Eprune_low_magnitude_flatten/assert_greater_equal/range/delta:output:0*
_output_shapes
: 28
6prune_low_magnitude_flatten/assert_greater_equal/range?
4prune_low_magnitude_flatten/assert_greater_equal/AllAllAprune_low_magnitude_flatten/assert_greater_equal/GreaterEqual:z:0?prune_low_magnitude_flatten/assert_greater_equal/range:output:0*
_output_shapes
: 26
4prune_low_magnitude_flatten/assert_greater_equal/All?
=prune_low_magnitude_flatten/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*?
value?B? B?Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2?
=prune_low_magnitude_flatten/assert_greater_equal/Assert/Const?
?prune_low_magnitude_flatten/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2A
?prune_low_magnitude_flatten/assert_greater_equal/Assert/Const_1?
?prune_low_magnitude_flatten/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*Y
valuePBN BHx (prune_low_magnitude_flatten/assert_greater_equal/ReadVariableOp:0) = 2A
?prune_low_magnitude_flatten/assert_greater_equal/Assert/Const_2?
?prune_low_magnitude_flatten/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*L
valueCBA B;y (prune_low_magnitude_flatten/assert_greater_equal/y:0) = 2A
?prune_low_magnitude_flatten/assert_greater_equal/Assert/Const_3?
Cprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuardIf=prune_low_magnitude_flatten/assert_greater_equal/All:output:0=prune_low_magnitude_flatten/assert_greater_equal/All:output:0Gprune_low_magnitude_flatten/assert_greater_equal/ReadVariableOp:value:0;prune_low_magnitude_flatten/assert_greater_equal/y:output:0J^prune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *c
else_branchTRR
Pprune_low_magnitude_flatten_assert_greater_equal_Assert_AssertGuard_false_247280*
output_shapes
: *b
then_branchSRQ
Oprune_low_magnitude_flatten_assert_greater_equal_Assert_AssertGuard_true_2472792E
Cprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard?
Lprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/IdentityIdentityLprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 2N
Lprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity?
7prune_low_magnitude_flatten/GreaterEqual/ReadVariableOpReadVariableOpHprune_low_magnitude_flatten_assert_greater_equal_readvariableop_resourceM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	29
7prune_low_magnitude_flatten/GreaterEqual/ReadVariableOp?
*prune_low_magnitude_flatten/GreaterEqual/yConstM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2,
*prune_low_magnitude_flatten/GreaterEqual/y?
(prune_low_magnitude_flatten/GreaterEqualGreaterEqual?prune_low_magnitude_flatten/GreaterEqual/ReadVariableOp:value:03prune_low_magnitude_flatten/GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2*
(prune_low_magnitude_flatten/GreaterEqual?
4prune_low_magnitude_flatten/LessEqual/ReadVariableOpReadVariableOpHprune_low_magnitude_flatten_assert_greater_equal_readvariableop_resourceM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	26
4prune_low_magnitude_flatten/LessEqual/ReadVariableOp?
'prune_low_magnitude_flatten/LessEqual/yConstM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2)
'prune_low_magnitude_flatten/LessEqual/y?
%prune_low_magnitude_flatten/LessEqual	LessEqual<prune_low_magnitude_flatten/LessEqual/ReadVariableOp:value:00prune_low_magnitude_flatten/LessEqual/y:output:0*
T0	*
_output_shapes
: 2'
%prune_low_magnitude_flatten/LessEqual?
"prune_low_magnitude_flatten/Less/xConstM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB :
?????????2$
"prune_low_magnitude_flatten/Less/x?
"prune_low_magnitude_flatten/Less/yConstM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : 2$
"prune_low_magnitude_flatten/Less/y?
 prune_low_magnitude_flatten/LessLess+prune_low_magnitude_flatten/Less/x:output:0+prune_low_magnitude_flatten/Less/y:output:0*
T0*
_output_shapes
: 2"
 prune_low_magnitude_flatten/Less?
%prune_low_magnitude_flatten/LogicalOr	LogicalOr)prune_low_magnitude_flatten/LessEqual:z:0$prune_low_magnitude_flatten/Less:z:0*
_output_shapes
: 2'
%prune_low_magnitude_flatten/LogicalOr?
&prune_low_magnitude_flatten/LogicalAnd
LogicalAnd,prune_low_magnitude_flatten/GreaterEqual:z:0)prune_low_magnitude_flatten/LogicalOr:z:0*
_output_shapes
: 2(
&prune_low_magnitude_flatten/LogicalAnd?
.prune_low_magnitude_flatten/Sub/ReadVariableOpReadVariableOpHprune_low_magnitude_flatten_assert_greater_equal_readvariableop_resourceM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	20
.prune_low_magnitude_flatten/Sub/ReadVariableOp?
!prune_low_magnitude_flatten/Sub/yConstM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2#
!prune_low_magnitude_flatten/Sub/y?
prune_low_magnitude_flatten/SubSub6prune_low_magnitude_flatten/Sub/ReadVariableOp:value:0*prune_low_magnitude_flatten/Sub/y:output:0*
T0	*
_output_shapes
: 2!
prune_low_magnitude_flatten/Sub?
&prune_low_magnitude_flatten/FloorMod/yConstM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd2(
&prune_low_magnitude_flatten/FloorMod/y?
$prune_low_magnitude_flatten/FloorModFloorMod#prune_low_magnitude_flatten/Sub:z:0/prune_low_magnitude_flatten/FloorMod/y:output:0*
T0	*
_output_shapes
: 2&
$prune_low_magnitude_flatten/FloorMod?
#prune_low_magnitude_flatten/Equal/yConstM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2%
#prune_low_magnitude_flatten/Equal/y?
!prune_low_magnitude_flatten/EqualEqual(prune_low_magnitude_flatten/FloorMod:z:0,prune_low_magnitude_flatten/Equal/y:output:0*
T0	*
_output_shapes
: 2#
!prune_low_magnitude_flatten/Equal?
(prune_low_magnitude_flatten/LogicalAnd_1
LogicalAnd*prune_low_magnitude_flatten/LogicalAnd:z:0%prune_low_magnitude_flatten/Equal:z:0*
_output_shapes
: 2*
(prune_low_magnitude_flatten/LogicalAnd_1?
!prune_low_magnitude_flatten/ConstConstM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *   ?2#
!prune_low_magnitude_flatten/Const?
 prune_low_magnitude_flatten/condStatelessIf,prune_low_magnitude_flatten/LogicalAnd_1:z:0,prune_low_magnitude_flatten/LogicalAnd_1:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *@
else_branch1R/
-prune_low_magnitude_flatten_cond_false_247320*
output_shapes
: *?
then_branch0R.
,prune_low_magnitude_flatten_cond_true_2473192"
 prune_low_magnitude_flatten/cond?
)prune_low_magnitude_flatten/cond/IdentityIdentity)prune_low_magnitude_flatten/cond:output:0*
T0
*
_output_shapes
: 2+
)prune_low_magnitude_flatten/cond/Identity?
"prune_low_magnitude_flatten/updateNoOpM^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity*^prune_low_magnitude_flatten/cond/Identity*
_output_shapes
 2$
"prune_low_magnitude_flatten/updaten
&prune_low_magnitude_flatten/group_depsNoOp*
_output_shapes
 2(
&prune_low_magnitude_flatten/group_deps?
#prune_low_magnitude_flatten/Const_1Const*
_output_shapes
:*
dtype0*
valueB"?????  2%
#prune_low_magnitude_flatten/Const_1?
#prune_low_magnitude_flatten/ReshapeReshapedropout/dropout/Mul_1:z:0,prune_low_magnitude_flatten/Const_1:output:0*
T0*(
_output_shapes
:??????????2%
#prune_low_magnitude_flatten/Reshape?
=prune_low_magnitude_dense/assert_greater_equal/ReadVariableOpReadVariableOpFprune_low_magnitude_dense_assert_greater_equal_readvariableop_resource*
_output_shapes
: *
dtype0	2?
=prune_low_magnitude_dense/assert_greater_equal/ReadVariableOp?
0prune_low_magnitude_dense/assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 22
0prune_low_magnitude_dense/assert_greater_equal/y?
;prune_low_magnitude_dense/assert_greater_equal/GreaterEqualGreaterEqualEprune_low_magnitude_dense/assert_greater_equal/ReadVariableOp:value:09prune_low_magnitude_dense/assert_greater_equal/y:output:0*
T0	*
_output_shapes
: 2=
;prune_low_magnitude_dense/assert_greater_equal/GreaterEqual?
3prune_low_magnitude_dense/assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 25
3prune_low_magnitude_dense/assert_greater_equal/Rank?
:prune_low_magnitude_dense/assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2<
:prune_low_magnitude_dense/assert_greater_equal/range/start?
:prune_low_magnitude_dense/assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2<
:prune_low_magnitude_dense/assert_greater_equal/range/delta?
4prune_low_magnitude_dense/assert_greater_equal/rangeRangeCprune_low_magnitude_dense/assert_greater_equal/range/start:output:0<prune_low_magnitude_dense/assert_greater_equal/Rank:output:0Cprune_low_magnitude_dense/assert_greater_equal/range/delta:output:0*
_output_shapes
: 26
4prune_low_magnitude_dense/assert_greater_equal/range?
2prune_low_magnitude_dense/assert_greater_equal/AllAll?prune_low_magnitude_dense/assert_greater_equal/GreaterEqual:z:0=prune_low_magnitude_dense/assert_greater_equal/range:output:0*
_output_shapes
: 24
2prune_low_magnitude_dense/assert_greater_equal/All?
;prune_low_magnitude_dense/assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*?
value?B? B?Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2=
;prune_low_magnitude_dense/assert_greater_equal/Assert/Const?
=prune_low_magnitude_dense/assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2?
=prune_low_magnitude_dense/assert_greater_equal/Assert/Const_1?
=prune_low_magnitude_dense/assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (prune_low_magnitude_dense/assert_greater_equal/ReadVariableOp:0) = 2?
=prune_low_magnitude_dense/assert_greater_equal/Assert/Const_2?
=prune_low_magnitude_dense/assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*J
valueAB? B9y (prune_low_magnitude_dense/assert_greater_equal/y:0) = 2?
=prune_low_magnitude_dense/assert_greater_equal/Assert/Const_3?
Aprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuardIf;prune_low_magnitude_dense/assert_greater_equal/All:output:0;prune_low_magnitude_dense/assert_greater_equal/All:output:0Eprune_low_magnitude_dense/assert_greater_equal/ReadVariableOp:value:09prune_low_magnitude_dense/assert_greater_equal/y:output:0D^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *a
else_branchRRP
Nprune_low_magnitude_dense_assert_greater_equal_Assert_AssertGuard_false_247347*
output_shapes
: *`
then_branchQRO
Mprune_low_magnitude_dense_assert_greater_equal_Assert_AssertGuard_true_2473462C
Aprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard?
Jprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/IdentityIdentityJprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 2L
Jprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity?
5prune_low_magnitude_dense/GreaterEqual/ReadVariableOpReadVariableOpFprune_low_magnitude_dense_assert_greater_equal_readvariableop_resourceK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	27
5prune_low_magnitude_dense/GreaterEqual/ReadVariableOp?
(prune_low_magnitude_dense/GreaterEqual/yConstK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2*
(prune_low_magnitude_dense/GreaterEqual/y?
&prune_low_magnitude_dense/GreaterEqualGreaterEqual=prune_low_magnitude_dense/GreaterEqual/ReadVariableOp:value:01prune_low_magnitude_dense/GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2(
&prune_low_magnitude_dense/GreaterEqual?
2prune_low_magnitude_dense/LessEqual/ReadVariableOpReadVariableOpFprune_low_magnitude_dense_assert_greater_equal_readvariableop_resourceK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	24
2prune_low_magnitude_dense/LessEqual/ReadVariableOp?
%prune_low_magnitude_dense/LessEqual/yConstK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2'
%prune_low_magnitude_dense/LessEqual/y?
#prune_low_magnitude_dense/LessEqual	LessEqual:prune_low_magnitude_dense/LessEqual/ReadVariableOp:value:0.prune_low_magnitude_dense/LessEqual/y:output:0*
T0	*
_output_shapes
: 2%
#prune_low_magnitude_dense/LessEqual?
 prune_low_magnitude_dense/Less/xConstK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB :
?????????2"
 prune_low_magnitude_dense/Less/x?
 prune_low_magnitude_dense/Less/yConstK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : 2"
 prune_low_magnitude_dense/Less/y?
prune_low_magnitude_dense/LessLess)prune_low_magnitude_dense/Less/x:output:0)prune_low_magnitude_dense/Less/y:output:0*
T0*
_output_shapes
: 2 
prune_low_magnitude_dense/Less?
#prune_low_magnitude_dense/LogicalOr	LogicalOr'prune_low_magnitude_dense/LessEqual:z:0"prune_low_magnitude_dense/Less:z:0*
_output_shapes
: 2%
#prune_low_magnitude_dense/LogicalOr?
$prune_low_magnitude_dense/LogicalAnd
LogicalAnd*prune_low_magnitude_dense/GreaterEqual:z:0'prune_low_magnitude_dense/LogicalOr:z:0*
_output_shapes
: 2&
$prune_low_magnitude_dense/LogicalAnd?
,prune_low_magnitude_dense/Sub/ReadVariableOpReadVariableOpFprune_low_magnitude_dense_assert_greater_equal_readvariableop_resourceK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2.
,prune_low_magnitude_dense/Sub/ReadVariableOp?
prune_low_magnitude_dense/Sub/yConstK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2!
prune_low_magnitude_dense/Sub/y?
prune_low_magnitude_dense/SubSub4prune_low_magnitude_dense/Sub/ReadVariableOp:value:0(prune_low_magnitude_dense/Sub/y:output:0*
T0	*
_output_shapes
: 2
prune_low_magnitude_dense/Sub?
$prune_low_magnitude_dense/FloorMod/yConstK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd2&
$prune_low_magnitude_dense/FloorMod/y?
"prune_low_magnitude_dense/FloorModFloorMod!prune_low_magnitude_dense/Sub:z:0-prune_low_magnitude_dense/FloorMod/y:output:0*
T0	*
_output_shapes
: 2$
"prune_low_magnitude_dense/FloorMod?
!prune_low_magnitude_dense/Equal/yConstK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2#
!prune_low_magnitude_dense/Equal/y?
prune_low_magnitude_dense/EqualEqual&prune_low_magnitude_dense/FloorMod:z:0*prune_low_magnitude_dense/Equal/y:output:0*
T0	*
_output_shapes
: 2!
prune_low_magnitude_dense/Equal?
&prune_low_magnitude_dense/LogicalAnd_1
LogicalAnd(prune_low_magnitude_dense/LogicalAnd:z:0#prune_low_magnitude_dense/Equal:z:0*
_output_shapes
: 2(
&prune_low_magnitude_dense/LogicalAnd_1?
prune_low_magnitude_dense/ConstConstK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *   ?2!
prune_low_magnitude_dense/Const?
prune_low_magnitude_dense/condIf*prune_low_magnitude_dense/LogicalAnd_1:z:0Fprune_low_magnitude_dense_assert_greater_equal_readvariableop_resource&prune_low_magnitude_dense_cond_input_1&prune_low_magnitude_dense_cond_input_2&prune_low_magnitude_dense_cond_input_3*prune_low_magnitude_dense/LogicalAnd_1:z:0*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: *$
_read_only_resource_inputs
*>
else_branch/R-
+prune_low_magnitude_dense_cond_false_247387*
output_shapes
: *=
then_branch.R,
*prune_low_magnitude_dense_cond_true_2473862 
prune_low_magnitude_dense/cond?
'prune_low_magnitude_dense/cond/IdentityIdentity'prune_low_magnitude_dense/cond:output:0*
T0
*
_output_shapes
: 2)
'prune_low_magnitude_dense/cond/Identity?
 prune_low_magnitude_dense/updateNoOpK^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity(^prune_low_magnitude_dense/cond/Identity*
_output_shapes
 2"
 prune_low_magnitude_dense/update?
,prune_low_magnitude_dense/Mul/ReadVariableOpReadVariableOp&prune_low_magnitude_dense_cond_input_1*
_output_shapes
:	?*
dtype02.
,prune_low_magnitude_dense/Mul/ReadVariableOp?
.prune_low_magnitude_dense/Mul/ReadVariableOp_1ReadVariableOp&prune_low_magnitude_dense_cond_input_2^prune_low_magnitude_dense/cond*
_output_shapes
:	?*
dtype020
.prune_low_magnitude_dense/Mul/ReadVariableOp_1?
prune_low_magnitude_dense/MulMul4prune_low_magnitude_dense/Mul/ReadVariableOp:value:06prune_low_magnitude_dense/Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	?2
prune_low_magnitude_dense/Mul?
*prune_low_magnitude_dense/AssignVariableOpAssignVariableOp&prune_low_magnitude_dense_cond_input_1!prune_low_magnitude_dense/Mul:z:0-^prune_low_magnitude_dense/Mul/ReadVariableOp^prune_low_magnitude_dense/cond*
_output_shapes
 *
dtype02,
*prune_low_magnitude_dense/AssignVariableOp?
$prune_low_magnitude_dense/group_depsNoOp+^prune_low_magnitude_dense/AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2&
$prune_low_magnitude_dense/group_deps?
&prune_low_magnitude_dense/group_deps_1NoOp%^prune_low_magnitude_dense/group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2(
&prune_low_magnitude_dense/group_deps_1?
/prune_low_magnitude_dense/MatMul/ReadVariableOpReadVariableOp&prune_low_magnitude_dense_cond_input_1+^prune_low_magnitude_dense/AssignVariableOp*
_output_shapes
:	?*
dtype021
/prune_low_magnitude_dense/MatMul/ReadVariableOp?
 prune_low_magnitude_dense/MatMulMatMul,prune_low_magnitude_flatten/Reshape:output:07prune_low_magnitude_dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2"
 prune_low_magnitude_dense/MatMul?
0prune_low_magnitude_dense/BiasAdd/ReadVariableOpReadVariableOp9prune_low_magnitude_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype022
0prune_low_magnitude_dense/BiasAdd/ReadVariableOp?
!prune_low_magnitude_dense/BiasAddBiasAdd*prune_low_magnitude_dense/MatMul:product:08prune_low_magnitude_dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2#
!prune_low_magnitude_dense/BiasAdd?
IdentityIdentity*prune_low_magnitude_dense/BiasAdd:output:0%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp,^prune_low_magnitude_conv2d/AssignVariableOp2^prune_low_magnitude_conv2d/BiasAdd/ReadVariableOp1^prune_low_magnitude_conv2d/Conv2D/ReadVariableOp7^prune_low_magnitude_conv2d/GreaterEqual/ReadVariableOp4^prune_low_magnitude_conv2d/LessEqual/ReadVariableOp.^prune_low_magnitude_conv2d/Mul/ReadVariableOp0^prune_low_magnitude_conv2d/Mul/ReadVariableOp_1.^prune_low_magnitude_conv2d/Sub/ReadVariableOpC^prune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard?^prune_low_magnitude_conv2d/assert_greater_equal/ReadVariableOp ^prune_low_magnitude_conv2d/cond+^prune_low_magnitude_dense/AssignVariableOp1^prune_low_magnitude_dense/BiasAdd/ReadVariableOp6^prune_low_magnitude_dense/GreaterEqual/ReadVariableOp3^prune_low_magnitude_dense/LessEqual/ReadVariableOp0^prune_low_magnitude_dense/MatMul/ReadVariableOp-^prune_low_magnitude_dense/Mul/ReadVariableOp/^prune_low_magnitude_dense/Mul/ReadVariableOp_1-^prune_low_magnitude_dense/Sub/ReadVariableOpB^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard>^prune_low_magnitude_dense/assert_greater_equal/ReadVariableOp^prune_low_magnitude_dense/cond8^prune_low_magnitude_flatten/GreaterEqual/ReadVariableOp5^prune_low_magnitude_flatten/LessEqual/ReadVariableOp/^prune_low_magnitude_flatten/Sub/ReadVariableOpD^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard@^prune_low_magnitude_flatten/assert_greater_equal/ReadVariableOp>^prune_low_magnitude_max_pooling2d/GreaterEqual/ReadVariableOp;^prune_low_magnitude_max_pooling2d/LessEqual/ReadVariableOp5^prune_low_magnitude_max_pooling2d/Sub/ReadVariableOpJ^prune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuardF^prune_low_magnitude_max_pooling2d/assert_greater_equal/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????(1: : : : : : : : : : : : : : 2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp2Z
+prune_low_magnitude_conv2d/AssignVariableOp+prune_low_magnitude_conv2d/AssignVariableOp2f
1prune_low_magnitude_conv2d/BiasAdd/ReadVariableOp1prune_low_magnitude_conv2d/BiasAdd/ReadVariableOp2d
0prune_low_magnitude_conv2d/Conv2D/ReadVariableOp0prune_low_magnitude_conv2d/Conv2D/ReadVariableOp2p
6prune_low_magnitude_conv2d/GreaterEqual/ReadVariableOp6prune_low_magnitude_conv2d/GreaterEqual/ReadVariableOp2j
3prune_low_magnitude_conv2d/LessEqual/ReadVariableOp3prune_low_magnitude_conv2d/LessEqual/ReadVariableOp2^
-prune_low_magnitude_conv2d/Mul/ReadVariableOp-prune_low_magnitude_conv2d/Mul/ReadVariableOp2b
/prune_low_magnitude_conv2d/Mul/ReadVariableOp_1/prune_low_magnitude_conv2d/Mul/ReadVariableOp_12^
-prune_low_magnitude_conv2d/Sub/ReadVariableOp-prune_low_magnitude_conv2d/Sub/ReadVariableOp2?
Bprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuardBprune_low_magnitude_conv2d/assert_greater_equal/Assert/AssertGuard2?
>prune_low_magnitude_conv2d/assert_greater_equal/ReadVariableOp>prune_low_magnitude_conv2d/assert_greater_equal/ReadVariableOp2B
prune_low_magnitude_conv2d/condprune_low_magnitude_conv2d/cond2X
*prune_low_magnitude_dense/AssignVariableOp*prune_low_magnitude_dense/AssignVariableOp2d
0prune_low_magnitude_dense/BiasAdd/ReadVariableOp0prune_low_magnitude_dense/BiasAdd/ReadVariableOp2n
5prune_low_magnitude_dense/GreaterEqual/ReadVariableOp5prune_low_magnitude_dense/GreaterEqual/ReadVariableOp2h
2prune_low_magnitude_dense/LessEqual/ReadVariableOp2prune_low_magnitude_dense/LessEqual/ReadVariableOp2b
/prune_low_magnitude_dense/MatMul/ReadVariableOp/prune_low_magnitude_dense/MatMul/ReadVariableOp2\
,prune_low_magnitude_dense/Mul/ReadVariableOp,prune_low_magnitude_dense/Mul/ReadVariableOp2`
.prune_low_magnitude_dense/Mul/ReadVariableOp_1.prune_low_magnitude_dense/Mul/ReadVariableOp_12\
,prune_low_magnitude_dense/Sub/ReadVariableOp,prune_low_magnitude_dense/Sub/ReadVariableOp2?
Aprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuardAprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard2~
=prune_low_magnitude_dense/assert_greater_equal/ReadVariableOp=prune_low_magnitude_dense/assert_greater_equal/ReadVariableOp2@
prune_low_magnitude_dense/condprune_low_magnitude_dense/cond2r
7prune_low_magnitude_flatten/GreaterEqual/ReadVariableOp7prune_low_magnitude_flatten/GreaterEqual/ReadVariableOp2l
4prune_low_magnitude_flatten/LessEqual/ReadVariableOp4prune_low_magnitude_flatten/LessEqual/ReadVariableOp2`
.prune_low_magnitude_flatten/Sub/ReadVariableOp.prune_low_magnitude_flatten/Sub/ReadVariableOp2?
Cprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuardCprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard2?
?prune_low_magnitude_flatten/assert_greater_equal/ReadVariableOp?prune_low_magnitude_flatten/assert_greater_equal/ReadVariableOp2~
=prune_low_magnitude_max_pooling2d/GreaterEqual/ReadVariableOp=prune_low_magnitude_max_pooling2d/GreaterEqual/ReadVariableOp2x
:prune_low_magnitude_max_pooling2d/LessEqual/ReadVariableOp:prune_low_magnitude_max_pooling2d/LessEqual/ReadVariableOp2l
4prune_low_magnitude_max_pooling2d/Sub/ReadVariableOp4prune_low_magnitude_max_pooling2d/Sub/ReadVariableOp2?
Iprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuardIprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard2?
Eprune_low_magnitude_max_pooling2d/assert_greater_equal/ReadVariableOpEprune_low_magnitude_max_pooling2d/assert_greater_equal/ReadVariableOp:W S
/
_output_shapes
:?????????(1
 
_user_specified_nameinputs
?[
?
*prune_low_magnitude_dense_cond_true_247386M
Cprune_low_magnitude_dense_cond_greaterequal_readvariableop_resource:	 Y
Fprune_low_magnitude_dense_cond_pruning_ops_abs_readvariableop_resource:	?K
8prune_low_magnitude_dense_cond_assignvariableop_resource:	?D
:prune_low_magnitude_dense_cond_assignvariableop_1_resource: R
Nprune_low_magnitude_dense_cond_identity_prune_low_magnitude_dense_logicaland_1
-
)prune_low_magnitude_dense_cond_identity_1
??/prune_low_magnitude_dense/cond/AssignVariableOp?1prune_low_magnitude_dense/cond/AssignVariableOp_1?:prune_low_magnitude_dense/cond/GreaterEqual/ReadVariableOp?7prune_low_magnitude_dense/cond/LessEqual/ReadVariableOp?1prune_low_magnitude_dense/cond/Sub/ReadVariableOp?=prune_low_magnitude_dense/cond/pruning_ops/Abs/ReadVariableOp?
:prune_low_magnitude_dense/cond/GreaterEqual/ReadVariableOpReadVariableOpCprune_low_magnitude_dense_cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	2<
:prune_low_magnitude_dense/cond/GreaterEqual/ReadVariableOp?
-prune_low_magnitude_dense/cond/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2/
-prune_low_magnitude_dense/cond/GreaterEqual/y?
+prune_low_magnitude_dense/cond/GreaterEqualGreaterEqualBprune_low_magnitude_dense/cond/GreaterEqual/ReadVariableOp:value:06prune_low_magnitude_dense/cond/GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2-
+prune_low_magnitude_dense/cond/GreaterEqual?
7prune_low_magnitude_dense/cond/LessEqual/ReadVariableOpReadVariableOpCprune_low_magnitude_dense_cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	29
7prune_low_magnitude_dense/cond/LessEqual/ReadVariableOp?
*prune_low_magnitude_dense/cond/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2,
*prune_low_magnitude_dense/cond/LessEqual/y?
(prune_low_magnitude_dense/cond/LessEqual	LessEqual?prune_low_magnitude_dense/cond/LessEqual/ReadVariableOp:value:03prune_low_magnitude_dense/cond/LessEqual/y:output:0*
T0	*
_output_shapes
: 2*
(prune_low_magnitude_dense/cond/LessEqual?
%prune_low_magnitude_dense/cond/Less/xConst*
_output_shapes
: *
dtype0*
valueB :
?????????2'
%prune_low_magnitude_dense/cond/Less/x?
%prune_low_magnitude_dense/cond/Less/yConst*
_output_shapes
: *
dtype0*
value	B : 2'
%prune_low_magnitude_dense/cond/Less/y?
#prune_low_magnitude_dense/cond/LessLess.prune_low_magnitude_dense/cond/Less/x:output:0.prune_low_magnitude_dense/cond/Less/y:output:0*
T0*
_output_shapes
: 2%
#prune_low_magnitude_dense/cond/Less?
(prune_low_magnitude_dense/cond/LogicalOr	LogicalOr,prune_low_magnitude_dense/cond/LessEqual:z:0'prune_low_magnitude_dense/cond/Less:z:0*
_output_shapes
: 2*
(prune_low_magnitude_dense/cond/LogicalOr?
)prune_low_magnitude_dense/cond/LogicalAnd
LogicalAnd/prune_low_magnitude_dense/cond/GreaterEqual:z:0,prune_low_magnitude_dense/cond/LogicalOr:z:0*
_output_shapes
: 2+
)prune_low_magnitude_dense/cond/LogicalAnd?
1prune_low_magnitude_dense/cond/Sub/ReadVariableOpReadVariableOpCprune_low_magnitude_dense_cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	23
1prune_low_magnitude_dense/cond/Sub/ReadVariableOp?
$prune_low_magnitude_dense/cond/Sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2&
$prune_low_magnitude_dense/cond/Sub/y?
"prune_low_magnitude_dense/cond/SubSub9prune_low_magnitude_dense/cond/Sub/ReadVariableOp:value:0-prune_low_magnitude_dense/cond/Sub/y:output:0*
T0	*
_output_shapes
: 2$
"prune_low_magnitude_dense/cond/Sub?
)prune_low_magnitude_dense/cond/FloorMod/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rd2+
)prune_low_magnitude_dense/cond/FloorMod/y?
'prune_low_magnitude_dense/cond/FloorModFloorMod&prune_low_magnitude_dense/cond/Sub:z:02prune_low_magnitude_dense/cond/FloorMod/y:output:0*
T0	*
_output_shapes
: 2)
'prune_low_magnitude_dense/cond/FloorMod?
&prune_low_magnitude_dense/cond/Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2(
&prune_low_magnitude_dense/cond/Equal/y?
$prune_low_magnitude_dense/cond/EqualEqual+prune_low_magnitude_dense/cond/FloorMod:z:0/prune_low_magnitude_dense/cond/Equal/y:output:0*
T0	*
_output_shapes
: 2&
$prune_low_magnitude_dense/cond/Equal?
+prune_low_magnitude_dense/cond/LogicalAnd_1
LogicalAnd-prune_low_magnitude_dense/cond/LogicalAnd:z:0(prune_low_magnitude_dense/cond/Equal:z:0*
_output_shapes
: 2-
+prune_low_magnitude_dense/cond/LogicalAnd_1?
$prune_low_magnitude_dense/cond/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2&
$prune_low_magnitude_dense/cond/Const?
=prune_low_magnitude_dense/cond/pruning_ops/Abs/ReadVariableOpReadVariableOpFprune_low_magnitude_dense_cond_pruning_ops_abs_readvariableop_resource*
_output_shapes
:	?*
dtype02?
=prune_low_magnitude_dense/cond/pruning_ops/Abs/ReadVariableOp?
.prune_low_magnitude_dense/cond/pruning_ops/AbsAbsEprune_low_magnitude_dense/cond/pruning_ops/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?20
.prune_low_magnitude_dense/cond/pruning_ops/Abs?
/prune_low_magnitude_dense/cond/pruning_ops/SizeConst*
_output_shapes
: *
dtype0*
value
B :?21
/prune_low_magnitude_dense/cond/pruning_ops/Size?
/prune_low_magnitude_dense/cond/pruning_ops/CastCast8prune_low_magnitude_dense/cond/pruning_ops/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: 21
/prune_low_magnitude_dense/cond/pruning_ops/Cast?
0prune_low_magnitude_dense/cond/pruning_ops/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??22
0prune_low_magnitude_dense/cond/pruning_ops/sub/x?
.prune_low_magnitude_dense/cond/pruning_ops/subSub9prune_low_magnitude_dense/cond/pruning_ops/sub/x:output:0-prune_low_magnitude_dense/cond/Const:output:0*
T0*
_output_shapes
: 20
.prune_low_magnitude_dense/cond/pruning_ops/sub?
.prune_low_magnitude_dense/cond/pruning_ops/mulMul3prune_low_magnitude_dense/cond/pruning_ops/Cast:y:02prune_low_magnitude_dense/cond/pruning_ops/sub:z:0*
T0*
_output_shapes
: 20
.prune_low_magnitude_dense/cond/pruning_ops/mul?
0prune_low_magnitude_dense/cond/pruning_ops/RoundRound2prune_low_magnitude_dense/cond/pruning_ops/mul:z:0*
T0*
_output_shapes
: 22
0prune_low_magnitude_dense/cond/pruning_ops/Round?
1prune_low_magnitude_dense/cond/pruning_ops/Cast_1Cast4prune_low_magnitude_dense/cond/pruning_ops/Round:y:0*

DstT0*

SrcT0*
_output_shapes
: 23
1prune_low_magnitude_dense/cond/pruning_ops/Cast_1?
8prune_low_magnitude_dense/cond/pruning_ops/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2:
8prune_low_magnitude_dense/cond/pruning_ops/Reshape/shape?
2prune_low_magnitude_dense/cond/pruning_ops/ReshapeReshape2prune_low_magnitude_dense/cond/pruning_ops/Abs:y:0Aprune_low_magnitude_dense/cond/pruning_ops/Reshape/shape:output:0*
T0*
_output_shapes	
:?24
2prune_low_magnitude_dense/cond/pruning_ops/Reshape?
1prune_low_magnitude_dense/cond/pruning_ops/Size_1Const*
_output_shapes
: *
dtype0*
value
B :?23
1prune_low_magnitude_dense/cond/pruning_ops/Size_1?
1prune_low_magnitude_dense/cond/pruning_ops/TopKV2TopKV2;prune_low_magnitude_dense/cond/pruning_ops/Reshape:output:0:prune_low_magnitude_dense/cond/pruning_ops/Size_1:output:0*
T0*"
_output_shapes
:?:?23
1prune_low_magnitude_dense/cond/pruning_ops/TopKV2?
2prune_low_magnitude_dense/cond/pruning_ops/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :24
2prune_low_magnitude_dense/cond/pruning_ops/sub_1/y?
0prune_low_magnitude_dense/cond/pruning_ops/sub_1Sub5prune_low_magnitude_dense/cond/pruning_ops/Cast_1:y:0;prune_low_magnitude_dense/cond/pruning_ops/sub_1/y:output:0*
T0*
_output_shapes
: 22
0prune_low_magnitude_dense/cond/pruning_ops/sub_1?
8prune_low_magnitude_dense/cond/pruning_ops/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2:
8prune_low_magnitude_dense/cond/pruning_ops/GatherV2/axis?
3prune_low_magnitude_dense/cond/pruning_ops/GatherV2GatherV2:prune_low_magnitude_dense/cond/pruning_ops/TopKV2:values:04prune_low_magnitude_dense/cond/pruning_ops/sub_1:z:0Aprune_low_magnitude_dense/cond/pruning_ops/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: 25
3prune_low_magnitude_dense/cond/pruning_ops/GatherV2?
7prune_low_magnitude_dense/cond/pruning_ops/GreaterEqualGreaterEqual2prune_low_magnitude_dense/cond/pruning_ops/Abs:y:0<prune_low_magnitude_dense/cond/pruning_ops/GatherV2:output:0*
T0*
_output_shapes
:	?29
7prune_low_magnitude_dense/cond/pruning_ops/GreaterEqual?
1prune_low_magnitude_dense/cond/pruning_ops/Cast_2Cast;prune_low_magnitude_dense/cond/pruning_ops/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?23
1prune_low_magnitude_dense/cond/pruning_ops/Cast_2?
/prune_low_magnitude_dense/cond/AssignVariableOpAssignVariableOp8prune_low_magnitude_dense_cond_assignvariableop_resource5prune_low_magnitude_dense/cond/pruning_ops/Cast_2:y:0*
_output_shapes
 *
dtype021
/prune_low_magnitude_dense/cond/AssignVariableOp?
1prune_low_magnitude_dense/cond/AssignVariableOp_1AssignVariableOp:prune_low_magnitude_dense_cond_assignvariableop_1_resource<prune_low_magnitude_dense/cond/pruning_ops/GatherV2:output:0*
_output_shapes
 *
dtype023
1prune_low_magnitude_dense/cond/AssignVariableOp_1?
)prune_low_magnitude_dense/cond/group_depsNoOp0^prune_low_magnitude_dense/cond/AssignVariableOp2^prune_low_magnitude_dense/cond/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2+
)prune_low_magnitude_dense/cond/group_deps?
'prune_low_magnitude_dense/cond/IdentityIdentityNprune_low_magnitude_dense_cond_identity_prune_low_magnitude_dense_logicaland_1*^prune_low_magnitude_dense/cond/group_deps*
T0
*
_output_shapes
: 2)
'prune_low_magnitude_dense/cond/Identity?
)prune_low_magnitude_dense/cond/Identity_1Identity0prune_low_magnitude_dense/cond/Identity:output:00^prune_low_magnitude_dense/cond/AssignVariableOp2^prune_low_magnitude_dense/cond/AssignVariableOp_1;^prune_low_magnitude_dense/cond/GreaterEqual/ReadVariableOp8^prune_low_magnitude_dense/cond/LessEqual/ReadVariableOp2^prune_low_magnitude_dense/cond/Sub/ReadVariableOp>^prune_low_magnitude_dense/cond/pruning_ops/Abs/ReadVariableOp*
T0
*
_output_shapes
: 2+
)prune_low_magnitude_dense/cond/Identity_1"_
)prune_low_magnitude_dense_cond_identity_12prune_low_magnitude_dense/cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2b
/prune_low_magnitude_dense/cond/AssignVariableOp/prune_low_magnitude_dense/cond/AssignVariableOp2f
1prune_low_magnitude_dense/cond/AssignVariableOp_11prune_low_magnitude_dense/cond/AssignVariableOp_12x
:prune_low_magnitude_dense/cond/GreaterEqual/ReadVariableOp:prune_low_magnitude_dense/cond/GreaterEqual/ReadVariableOp2r
7prune_low_magnitude_dense/cond/LessEqual/ReadVariableOp7prune_low_magnitude_dense/cond/LessEqual/ReadVariableOp2f
1prune_low_magnitude_dense/cond/Sub/ReadVariableOp1prune_low_magnitude_dense/cond/Sub/ReadVariableOp2~
=prune_low_magnitude_dense/cond/pruning_ops/Abs/ReadVariableOp=prune_low_magnitude_dense/cond/pruning_ops/Abs/ReadVariableOp:

_output_shapes
: 
?
?
,prune_low_magnitude_flatten_cond_true_247319V
Rprune_low_magnitude_flatten_cond_identity_prune_low_magnitude_flatten_logicaland_1
/
+prune_low_magnitude_flatten_cond_identity_1
x
+prune_low_magnitude_flatten/cond/group_depsNoOp*
_output_shapes
 2-
+prune_low_magnitude_flatten/cond/group_deps?
)prune_low_magnitude_flatten/cond/IdentityIdentityRprune_low_magnitude_flatten_cond_identity_prune_low_magnitude_flatten_logicaland_1,^prune_low_magnitude_flatten/cond/group_deps*
T0
*
_output_shapes
: 2+
)prune_low_magnitude_flatten/cond/Identity?
+prune_low_magnitude_flatten/cond/Identity_1Identity2prune_low_magnitude_flatten/cond/Identity:output:0*
T0
*
_output_shapes
: 2-
+prune_low_magnitude_flatten/cond/Identity_1"c
+prune_low_magnitude_flatten_cond_identity_14prune_low_magnitude_flatten/cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
+prune_low_magnitude_dense_cond_false_247387.
*prune_low_magnitude_dense_cond_placeholder0
,prune_low_magnitude_dense_cond_placeholder_10
,prune_low_magnitude_dense_cond_placeholder_20
,prune_low_magnitude_dense_cond_placeholder_3R
Nprune_low_magnitude_dense_cond_identity_prune_low_magnitude_dense_logicaland_1
-
)prune_low_magnitude_dense_cond_identity_1
h
#prune_low_magnitude_dense/cond/NoOpNoOp*
_output_shapes
 2%
#prune_low_magnitude_dense/cond/NoOp?
'prune_low_magnitude_dense/cond/IdentityIdentityNprune_low_magnitude_dense_cond_identity_prune_low_magnitude_dense_logicaland_1$^prune_low_magnitude_dense/cond/NoOp*
T0
*
_output_shapes
: 2)
'prune_low_magnitude_dense/cond/Identity?
)prune_low_magnitude_dense/cond/Identity_1Identity0prune_low_magnitude_dense/cond/Identity:output:0*
T0
*
_output_shapes
: 2+
)prune_low_magnitude_dense/cond/Identity_1"_
)prune_low_magnitude_dense_cond_identity_12prune_low_magnitude_dense/cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : :

_output_shapes
: 
?
?
4assert_greater_equal_Assert_AssertGuard_false_247708K
Gassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all
V
Rassert_greater_equal_assert_assertguard_assert_assert_greater_equal_readvariableop	I
Eassert_greater_equal_assert_assertguard_assert_assert_greater_equal_y	6
2assert_greater_equal_assert_assertguard_identity_1
??.assert_greater_equal/Assert/AssertGuard/Assert?
5assert_greater_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*?
value?B? B?Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.27
5assert_greater_equal/Assert/AssertGuard/Assert/data_0?
5assert_greater_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:27
5assert_greater_equal/Assert/AssertGuard/Assert/data_1?
5assert_greater_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 27
5assert_greater_equal/Assert/AssertGuard/Assert/data_2?
5assert_greater_equal/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 27
5assert_greater_equal/Assert/AssertGuard/Assert/data_4?
.assert_greater_equal/Assert/AssertGuard/AssertAssertGassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all>assert_greater_equal/Assert/AssertGuard/Assert/data_0:output:0>assert_greater_equal/Assert/AssertGuard/Assert/data_1:output:0>assert_greater_equal/Assert/AssertGuard/Assert/data_2:output:0Rassert_greater_equal_assert_assertguard_assert_assert_greater_equal_readvariableop>assert_greater_equal/Assert/AssertGuard/Assert/data_4:output:0Eassert_greater_equal_assert_assertguard_assert_assert_greater_equal_y*
T

2		*
_output_shapes
 20
.assert_greater_equal/Assert/AssertGuard/Assert?
0assert_greater_equal/Assert/AssertGuard/IdentityIdentityGassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all/^assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identity?
2assert_greater_equal/Assert/AssertGuard/Identity_1Identity9assert_greater_equal/Assert/AssertGuard/Identity:output:0/^assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 24
2assert_greater_equal/Assert/AssertGuard/Identity_1"q
2assert_greater_equal_assert_assertguard_identity_1;assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2`
.assert_greater_equal/Assert/AssertGuard/Assert.assert_greater_equal/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
b
C__inference_dropout_layer_call_and_return_conditional_losses_247788

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
?7
?
cond_true_2466273
)cond_greaterequal_readvariableop_resource:	 F
,cond_pruning_ops_abs_readvariableop_resource:
8
cond_assignvariableop_resource:
*
 cond_assignvariableop_1_resource: 
cond_identity_logicaland_1

cond_identity_1
??cond/AssignVariableOp?cond/AssignVariableOp_1? cond/GreaterEqual/ReadVariableOp?cond/LessEqual/ReadVariableOp?cond/Sub/ReadVariableOp?#cond/pruning_ops/Abs/ReadVariableOp?
 cond/GreaterEqual/ReadVariableOpReadVariableOp)cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	2"
 cond/GreaterEqual/ReadVariableOpl
cond/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
cond/GreaterEqual/y?
cond/GreaterEqualGreaterEqual(cond/GreaterEqual/ReadVariableOp:value:0cond/GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
cond/GreaterEqual?
cond/LessEqual/ReadVariableOpReadVariableOp)cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	2
cond/LessEqual/ReadVariableOpo
cond/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2
cond/LessEqual/y?
cond/LessEqual	LessEqual%cond/LessEqual/ReadVariableOp:value:0cond/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
cond/LessEquale
cond/Less/xConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
cond/Less/x\
cond/Less/yConst*
_output_shapes
: *
dtype0*
value	B : 2
cond/Less/yk
	cond/LessLesscond/Less/x:output:0cond/Less/y:output:0*
T0*
_output_shapes
: 2
	cond/Lessh
cond/LogicalOr	LogicalOrcond/LessEqual:z:0cond/Less:z:0*
_output_shapes
: 2
cond/LogicalOrs
cond/LogicalAnd
LogicalAndcond/GreaterEqual:z:0cond/LogicalOr:z:0*
_output_shapes
: 2
cond/LogicalAnd?
cond/Sub/ReadVariableOpReadVariableOp)cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	2
cond/Sub/ReadVariableOpZ

cond/Sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2

cond/Sub/yr
cond/SubSubcond/Sub/ReadVariableOp:value:0cond/Sub/y:output:0*
T0	*
_output_shapes
: 2

cond/Subd
cond/FloorMod/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rd2
cond/FloorMod/ys
cond/FloorModFloorModcond/Sub:z:0cond/FloorMod/y:output:0*
T0	*
_output_shapes
: 2
cond/FloorMod^
cond/Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
cond/Equal/yl

cond/EqualEqualcond/FloorMod:z:0cond/Equal/y:output:0*
T0	*
_output_shapes
: 2

cond/Equalq
cond/LogicalAnd_1
LogicalAndcond/LogicalAnd:z:0cond/Equal:z:0*
_output_shapes
: 2
cond/LogicalAnd_1]

cond/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

cond/Const?
#cond/pruning_ops/Abs/ReadVariableOpReadVariableOp,cond_pruning_ops_abs_readvariableop_resource*&
_output_shapes
:
*
dtype02%
#cond/pruning_ops/Abs/ReadVariableOp?
cond/pruning_ops/AbsAbs+cond/pruning_ops/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:
2
cond/pruning_ops/Absq
cond/pruning_ops/SizeConst*
_output_shapes
: *
dtype0*
value
B :?2
cond/pruning_ops/Size?
cond/pruning_ops/CastCastcond/pruning_ops/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
cond/pruning_ops/Castu
cond/pruning_ops/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
cond/pruning_ops/sub/x?
cond/pruning_ops/subSubcond/pruning_ops/sub/x:output:0cond/Const:output:0*
T0*
_output_shapes
: 2
cond/pruning_ops/sub?
cond/pruning_ops/mulMulcond/pruning_ops/Cast:y:0cond/pruning_ops/sub:z:0*
T0*
_output_shapes
: 2
cond/pruning_ops/mult
cond/pruning_ops/RoundRoundcond/pruning_ops/mul:z:0*
T0*
_output_shapes
: 2
cond/pruning_ops/Round?
cond/pruning_ops/Cast_1Castcond/pruning_ops/Round:y:0*

DstT0*

SrcT0*
_output_shapes
: 2
cond/pruning_ops/Cast_1?
cond/pruning_ops/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2 
cond/pruning_ops/Reshape/shape?
cond/pruning_ops/ReshapeReshapecond/pruning_ops/Abs:y:0'cond/pruning_ops/Reshape/shape:output:0*
T0*
_output_shapes	
:?2
cond/pruning_ops/Reshapeu
cond/pruning_ops/Size_1Const*
_output_shapes
: *
dtype0*
value
B :?2
cond/pruning_ops/Size_1?
cond/pruning_ops/TopKV2TopKV2!cond/pruning_ops/Reshape:output:0 cond/pruning_ops/Size_1:output:0*
T0*"
_output_shapes
:?:?2
cond/pruning_ops/TopKV2v
cond/pruning_ops/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
cond/pruning_ops/sub_1/y?
cond/pruning_ops/sub_1Subcond/pruning_ops/Cast_1:y:0!cond/pruning_ops/sub_1/y:output:0*
T0*
_output_shapes
: 2
cond/pruning_ops/sub_1?
cond/pruning_ops/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
cond/pruning_ops/GatherV2/axis?
cond/pruning_ops/GatherV2GatherV2 cond/pruning_ops/TopKV2:values:0cond/pruning_ops/sub_1:z:0'cond/pruning_ops/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: 2
cond/pruning_ops/GatherV2?
cond/pruning_ops/GreaterEqualGreaterEqualcond/pruning_ops/Abs:y:0"cond/pruning_ops/GatherV2:output:0*
T0*&
_output_shapes
:
2
cond/pruning_ops/GreaterEqual?
cond/pruning_ops/Cast_2Cast!cond/pruning_ops/GreaterEqual:z:0*

DstT0*

SrcT0
*&
_output_shapes
:
2
cond/pruning_ops/Cast_2?
cond/AssignVariableOpAssignVariableOpcond_assignvariableop_resourcecond/pruning_ops/Cast_2:y:0*
_output_shapes
 *
dtype02
cond/AssignVariableOp?
cond/AssignVariableOp_1AssignVariableOp cond_assignvariableop_1_resource"cond/pruning_ops/GatherV2:output:0*
_output_shapes
 *
dtype02
cond/AssignVariableOp_1?
cond/group_depsNoOp^cond/AssignVariableOp^cond/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2
cond/group_depsy
cond/IdentityIdentitycond_identity_logicaland_1^cond/group_deps*
T0
*
_output_shapes
: 2
cond/Identity?
cond/Identity_1Identitycond/Identity:output:0^cond/AssignVariableOp^cond/AssignVariableOp_1!^cond/GreaterEqual/ReadVariableOp^cond/LessEqual/ReadVariableOp^cond/Sub/ReadVariableOp$^cond/pruning_ops/Abs/ReadVariableOp*
T0
*
_output_shapes
: 2
cond/Identity_1"+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2.
cond/AssignVariableOpcond/AssignVariableOp22
cond/AssignVariableOp_1cond/AssignVariableOp_12D
 cond/GreaterEqual/ReadVariableOp cond/GreaterEqual/ReadVariableOp2>
cond/LessEqual/ReadVariableOpcond/LessEqual/ReadVariableOp22
cond/Sub/ReadVariableOpcond/Sub/ReadVariableOp2J
#cond/pruning_ops/Abs/ReadVariableOp#cond/pruning_ops/Abs/ReadVariableOp:

_output_shapes
: 
?
s
W__inference_prune_low_magnitude_flatten_layer_call_and_return_conditional_losses_246168

inputs
identity4
	no_updateNoOp*
_output_shapes
 2
	no_update6

group_depsNoOp*
_output_shapes
 2

group_deps_
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
?
?
4assert_greater_equal_Assert_AssertGuard_false_246499K
Gassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all
V
Rassert_greater_equal_assert_assertguard_assert_assert_greater_equal_readvariableop	I
Eassert_greater_equal_assert_assertguard_assert_assert_greater_equal_y	6
2assert_greater_equal_assert_assertguard_identity_1
??.assert_greater_equal/Assert/AssertGuard/Assert?
5assert_greater_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*?
value?B? B?Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.27
5assert_greater_equal/Assert/AssertGuard/Assert/data_0?
5assert_greater_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:27
5assert_greater_equal/Assert/AssertGuard/Assert/data_1?
5assert_greater_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 27
5assert_greater_equal/Assert/AssertGuard/Assert/data_2?
5assert_greater_equal/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 27
5assert_greater_equal/Assert/AssertGuard/Assert/data_4?
.assert_greater_equal/Assert/AssertGuard/AssertAssertGassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all>assert_greater_equal/Assert/AssertGuard/Assert/data_0:output:0>assert_greater_equal/Assert/AssertGuard/Assert/data_1:output:0>assert_greater_equal/Assert/AssertGuard/Assert/data_2:output:0Rassert_greater_equal_assert_assertguard_assert_assert_greater_equal_readvariableop>assert_greater_equal/Assert/AssertGuard/Assert/data_4:output:0Eassert_greater_equal_assert_assertguard_assert_assert_greater_equal_y*
T

2		*
_output_shapes
 20
.assert_greater_equal/Assert/AssertGuard/Assert?
0assert_greater_equal/Assert/AssertGuard/IdentityIdentityGassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all/^assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identity?
2assert_greater_equal/Assert/AssertGuard/Identity_1Identity9assert_greater_equal/Assert/AssertGuard/Identity:output:0/^assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 24
2assert_greater_equal/Assert/AssertGuard/Identity_1"q
2assert_greater_equal_assert_assertguard_identity_1;assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2`
.assert_greater_equal/Assert/AssertGuard/Assert.assert_greater_equal/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?	
?
3assert_greater_equal_Assert_AssertGuard_true_246498M
Iassert_greater_equal_assert_assertguard_identity_assert_greater_equal_all
7
3assert_greater_equal_assert_assertguard_placeholder	9
5assert_greater_equal_assert_assertguard_placeholder_1	6
2assert_greater_equal_assert_assertguard_identity_1
z
,assert_greater_equal/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2.
,assert_greater_equal/Assert/AssertGuard/NoOp?
0assert_greater_equal/Assert/AssertGuard/IdentityIdentityIassert_greater_equal_assert_assertguard_identity_assert_greater_equal_all-^assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identity?
2assert_greater_equal/Assert/AssertGuard/Identity_1Identity9assert_greater_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 24
2assert_greater_equal/Assert/AssertGuard/Identity_1"q
2assert_greater_equal_assert_assertguard_identity_1;assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?G
?
U__inference_prune_low_magnitude_dense_layer_call_and_return_conditional_losses_246354

inputs6
,assert_greater_equal_readvariableop_resource:	 
cond_input_1:	?
cond_input_2:	?
cond_input_3: -
biasadd_readvariableop_resource:
identity??AssignVariableOp?BiasAdd/ReadVariableOp?GreaterEqual/ReadVariableOp?LessEqual/ReadVariableOp?MatMul/ReadVariableOp?Mul/ReadVariableOp?Mul/ReadVariableOp_1?Sub/ReadVariableOp?'assert_greater_equal/Assert/AssertGuard?#assert_greater_equal/ReadVariableOp?cond?
#assert_greater_equal/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource*
_output_shapes
: *
dtype0	2%
#assert_greater_equal/ReadVariableOpr
assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
assert_greater_equal/y?
!assert_greater_equal/GreaterEqualGreaterEqual+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
T0	*
_output_shapes
: 2#
!assert_greater_equal/GreaterEqualx
assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 2
assert_greater_equal/Rank?
 assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2"
 assert_greater_equal/range/start?
 assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2"
 assert_greater_equal/range/delta?
assert_greater_equal/rangeRange)assert_greater_equal/range/start:output:0"assert_greater_equal/Rank:output:0)assert_greater_equal/range/delta:output:0*
_output_shapes
: 2
assert_greater_equal/range?
assert_greater_equal/AllAll%assert_greater_equal/GreaterEqual:z:0#assert_greater_equal/range:output:0*
_output_shapes
: 2
assert_greater_equal/All?
!assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*?
value?B? B?Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2#
!assert_greater_equal/Assert/Const?
#assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2%
#assert_greater_equal/Assert/Const_1?
#assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2%
#assert_greater_equal/Assert/Const_2?
#assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2%
#assert_greater_equal/Assert/Const_3?
'assert_greater_equal/Assert/AssertGuardIf!assert_greater_equal/All:output:0!assert_greater_equal/All:output:0+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *G
else_branch8R6
4assert_greater_equal_Assert_AssertGuard_false_246243*
output_shapes
: *F
then_branch7R5
3assert_greater_equal_Assert_AssertGuard_true_2462422)
'assert_greater_equal/Assert/AssertGuard?
0assert_greater_equal/Assert/AssertGuard/IdentityIdentity0assert_greater_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identity?
GreaterEqual/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOp?
GreaterEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y?
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqual?
LessEqual/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp?
LessEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqual?
Less/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB :
?????????2
Less/x?
Less/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd?
Sub/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOp?
Sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
Sub?

FloorMod/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorMod?
Equal/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1?
ConstConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Const?
condIfLogicalAnd_1:z:0,assert_greater_equal_readvariableop_resourcecond_input_1cond_input_2cond_input_3LogicalAnd_1:z:0*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: *$
_read_only_resource_inputs
*$
else_branchR
cond_false_246283*
output_shapes
: *#
then_branchR
cond_true_2462822
condZ
cond/IdentityIdentitycond:output:0*
T0
*
_output_shapes
: 2
cond/Identityq
updateNoOp1^assert_greater_equal/Assert/AssertGuard/Identity^cond/Identity*
_output_shapes
 2
updatev
Mul/ReadVariableOpReadVariableOpcond_input_1*
_output_shapes
:	?*
dtype02
Mul/ReadVariableOp?
Mul/ReadVariableOp_1ReadVariableOpcond_input_2^cond*
_output_shapes
:	?*
dtype02
Mul/ReadVariableOp_1u
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	?2
Mul?
AssignVariableOpAssignVariableOpcond_input_1Mul:z:0^Mul/ReadVariableOp^cond*
_output_shapes
 *
dtype02
AssignVariableOpw

group_depsNoOp^AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2

group_depsu
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2
group_deps_1?
MatMul/ReadVariableOpReadVariableOpcond_input_1^AssignVariableOp*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
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
BiasAdd?
IdentityIdentityBiasAdd:output:0^AssignVariableOp^BiasAdd/ReadVariableOp^GreaterEqual/ReadVariableOp^LessEqual/ReadVariableOp^MatMul/ReadVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1^Sub/ReadVariableOp(^assert_greater_equal/Assert/AssertGuard$^assert_greater_equal/ReadVariableOp^cond*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*1
_input_shapes 
:??????????: : : : : 2$
AssignVariableOpAssignVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2:
GreaterEqual/ReadVariableOpGreaterEqual/ReadVariableOp24
LessEqual/ReadVariableOpLessEqual/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_12(
Sub/ReadVariableOpSub/ReadVariableOp2R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard2J
#assert_greater_equal/ReadVariableOp#assert_greater_equal/ReadVariableOp2
condcond:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
3prune_low_magnitude_max_pooling2d_cond_false_247246b
^prune_low_magnitude_max_pooling2d_cond_identity_prune_low_magnitude_max_pooling2d_logicaland_1
5
1prune_low_magnitude_max_pooling2d_cond_identity_1
x
+prune_low_magnitude_max_pooling2d/cond/NoOpNoOp*
_output_shapes
 2-
+prune_low_magnitude_max_pooling2d/cond/NoOp?
/prune_low_magnitude_max_pooling2d/cond/IdentityIdentity^prune_low_magnitude_max_pooling2d_cond_identity_prune_low_magnitude_max_pooling2d_logicaland_1,^prune_low_magnitude_max_pooling2d/cond/NoOp*
T0
*
_output_shapes
: 21
/prune_low_magnitude_max_pooling2d/cond/Identity?
1prune_low_magnitude_max_pooling2d/cond/Identity_1Identity8prune_low_magnitude_max_pooling2d/cond/Identity:output:0*
T0
*
_output_shapes
: 23
1prune_low_magnitude_max_pooling2d/cond/Identity_1"o
1prune_low_magnitude_max_pooling2d_cond_identity_1:prune_low_magnitude_max_pooling2d/cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
?
Uprune_low_magnitude_max_pooling2d_assert_greater_equal_Assert_AssertGuard_true_247205?
?prune_low_magnitude_max_pooling2d_assert_greater_equal_assert_assertguard_identity_prune_low_magnitude_max_pooling2d_assert_greater_equal_all
Y
Uprune_low_magnitude_max_pooling2d_assert_greater_equal_assert_assertguard_placeholder	[
Wprune_low_magnitude_max_pooling2d_assert_greater_equal_assert_assertguard_placeholder_1	X
Tprune_low_magnitude_max_pooling2d_assert_greater_equal_assert_assertguard_identity_1
?
Nprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2P
Nprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/NoOp?
Rprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/IdentityIdentity?prune_low_magnitude_max_pooling2d_assert_greater_equal_assert_assertguard_identity_prune_low_magnitude_max_pooling2d_assert_greater_equal_allO^prune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 2T
Rprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Identity?
Tprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Identity_1Identity[prune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 2V
Tprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Identity_1"?
Tprune_low_magnitude_max_pooling2d_assert_greater_equal_assert_assertguard_identity_1]prune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
4assert_greater_equal_Assert_AssertGuard_false_247934K
Gassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all
V
Rassert_greater_equal_assert_assertguard_assert_assert_greater_equal_readvariableop	I
Eassert_greater_equal_assert_assertguard_assert_assert_greater_equal_y	6
2assert_greater_equal_assert_assertguard_identity_1
??.assert_greater_equal/Assert/AssertGuard/Assert?
5assert_greater_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*?
value?B? B?Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.27
5assert_greater_equal/Assert/AssertGuard/Assert/data_0?
5assert_greater_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:27
5assert_greater_equal/Assert/AssertGuard/Assert/data_1?
5assert_greater_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 27
5assert_greater_equal/Assert/AssertGuard/Assert/data_2?
5assert_greater_equal/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 27
5assert_greater_equal/Assert/AssertGuard/Assert/data_4?
.assert_greater_equal/Assert/AssertGuard/AssertAssertGassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all>assert_greater_equal/Assert/AssertGuard/Assert/data_0:output:0>assert_greater_equal/Assert/AssertGuard/Assert/data_1:output:0>assert_greater_equal/Assert/AssertGuard/Assert/data_2:output:0Rassert_greater_equal_assert_assertguard_assert_assert_greater_equal_readvariableop>assert_greater_equal/Assert/AssertGuard/Assert/data_4:output:0Eassert_greater_equal_assert_assertguard_assert_assert_greater_equal_y*
T

2		*
_output_shapes
 20
.assert_greater_equal/Assert/AssertGuard/Assert?
0assert_greater_equal/Assert/AssertGuard/IdentityIdentityGassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all/^assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identity?
2assert_greater_equal/Assert/AssertGuard/Identity_1Identity9assert_greater_equal/Assert/AssertGuard/Identity:output:0/^assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 24
2assert_greater_equal/Assert/AssertGuard/Identity_1"q
2assert_greater_equal_assert_assertguard_identity_1;assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2`
.assert_greater_equal/Assert/AssertGuard/Assert.assert_greater_equal/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?7
?	
Q__inference_ExtAudioDataModelPrun_layer_call_and_return_conditional_losses_246926
input_1;
-normalization_reshape_readvariableop_resource:=
/normalization_reshape_1_readvariableop_resource:+
!prune_low_magnitude_conv2d_246896:	 ;
!prune_low_magnitude_conv2d_246898:
;
!prune_low_magnitude_conv2d_246900:
+
!prune_low_magnitude_conv2d_246902: /
!prune_low_magnitude_conv2d_246904:2
(prune_low_magnitude_max_pooling2d_246907:	 ,
"prune_low_magnitude_flatten_246911:	 *
 prune_low_magnitude_dense_246914:	 3
 prune_low_magnitude_dense_246916:	?3
 prune_low_magnitude_dense_246918:	?*
 prune_low_magnitude_dense_246920: .
 prune_low_magnitude_dense_246922:
identity??dropout/StatefulPartitionedCall?$normalization/Reshape/ReadVariableOp?&normalization/Reshape_1/ReadVariableOp?2prune_low_magnitude_conv2d/StatefulPartitionedCall?1prune_low_magnitude_dense/StatefulPartitionedCall?3prune_low_magnitude_flatten/StatefulPartitionedCall?9prune_low_magnitude_max_pooling2d/StatefulPartitionedCall?
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp?
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape/shape?
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:2
normalization/Reshape?
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp?
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape_1/shape?
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
normalization/Reshape_1?
normalization/subSubinput_1normalization/Reshape:output:0*
T0*/
_output_shapes
:?????????(12
normalization/sub?
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*&
_output_shapes
:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*&
_output_shapes
:2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*/
_output_shapes
:?????????(12
normalization/truediv?
2prune_low_magnitude_conv2d/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0!prune_low_magnitude_conv2d_246896!prune_low_magnitude_conv2d_246898!prune_low_magnitude_conv2d_246900!prune_low_magnitude_conv2d_246902!prune_low_magnitude_conv2d_246904*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_prune_low_magnitude_conv2d_layer_call_and_return_conditional_losses_24670024
2prune_low_magnitude_conv2d/StatefulPartitionedCall?
9prune_low_magnitude_max_pooling2d/StatefulPartitionedCallStatefulPartitionedCall;prune_low_magnitude_conv2d/StatefulPartitionedCall:output:0(prune_low_magnitude_max_pooling2d_246907*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *f
faR_
]__inference_prune_low_magnitude_max_pooling2d_layer_call_and_return_conditional_losses_2465522;
9prune_low_magnitude_max_pooling2d/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCallBprune_low_magnitude_max_pooling2d/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_2464712!
dropout/StatefulPartitionedCall?
3prune_low_magnitude_flatten/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0"prune_low_magnitude_flatten_246911*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *`
f[RY
W__inference_prune_low_magnitude_flatten_layer_call_and_return_conditional_losses_24644625
3prune_low_magnitude_flatten/StatefulPartitionedCall?
1prune_low_magnitude_dense/StatefulPartitionedCallStatefulPartitionedCall<prune_low_magnitude_flatten/StatefulPartitionedCall:output:0 prune_low_magnitude_dense_246914 prune_low_magnitude_dense_246916 prune_low_magnitude_dense_246918 prune_low_magnitude_dense_246920 prune_low_magnitude_dense_246922*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_prune_low_magnitude_dense_layer_call_and_return_conditional_losses_24635423
1prune_low_magnitude_dense/StatefulPartitionedCall?
IdentityIdentity:prune_low_magnitude_dense/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp3^prune_low_magnitude_conv2d/StatefulPartitionedCall2^prune_low_magnitude_dense/StatefulPartitionedCall4^prune_low_magnitude_flatten/StatefulPartitionedCall:^prune_low_magnitude_max_pooling2d/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????(1: : : : : : : : : : : : : : 2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp2h
2prune_low_magnitude_conv2d/StatefulPartitionedCall2prune_low_magnitude_conv2d/StatefulPartitionedCall2f
1prune_low_magnitude_dense/StatefulPartitionedCall1prune_low_magnitude_dense/StatefulPartitionedCall2j
3prune_low_magnitude_flatten/StatefulPartitionedCall3prune_low_magnitude_flatten/StatefulPartitionedCall2v
9prune_low_magnitude_max_pooling2d/StatefulPartitionedCall9prune_low_magnitude_max_pooling2d/StatefulPartitionedCall:X T
/
_output_shapes
:?????????(1
!
_user_specified_name	input_1
?
?
Mprune_low_magnitude_dense_assert_greater_equal_Assert_AssertGuard_true_247346?
}prune_low_magnitude_dense_assert_greater_equal_assert_assertguard_identity_prune_low_magnitude_dense_assert_greater_equal_all
Q
Mprune_low_magnitude_dense_assert_greater_equal_assert_assertguard_placeholder	S
Oprune_low_magnitude_dense_assert_greater_equal_assert_assertguard_placeholder_1	P
Lprune_low_magnitude_dense_assert_greater_equal_assert_assertguard_identity_1
?
Fprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2H
Fprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/NoOp?
Jprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/IdentityIdentity}prune_low_magnitude_dense_assert_greater_equal_assert_assertguard_identity_prune_low_magnitude_dense_assert_greater_equal_allG^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 2L
Jprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity?
Lprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity_1IdentitySprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 2N
Lprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity_1"?
Lprune_low_magnitude_dense_assert_greater_equal_assert_assertguard_identity_1Uprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?I
?
V__inference_prune_low_magnitude_conv2d_layer_call_and_return_conditional_losses_247674

inputs6
,assert_greater_equal_readvariableop_resource:	 &
cond_input_1:
&
cond_input_2:

cond_input_3: -
biasadd_readvariableop_resource:
identity??AssignVariableOp?BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?GreaterEqual/ReadVariableOp?LessEqual/ReadVariableOp?Mul/ReadVariableOp?Mul/ReadVariableOp_1?Sub/ReadVariableOp?'assert_greater_equal/Assert/AssertGuard?#assert_greater_equal/ReadVariableOp?cond?
#assert_greater_equal/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource*
_output_shapes
: *
dtype0	2%
#assert_greater_equal/ReadVariableOpr
assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
assert_greater_equal/y?
!assert_greater_equal/GreaterEqualGreaterEqual+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
T0	*
_output_shapes
: 2#
!assert_greater_equal/GreaterEqualx
assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 2
assert_greater_equal/Rank?
 assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2"
 assert_greater_equal/range/start?
 assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2"
 assert_greater_equal/range/delta?
assert_greater_equal/rangeRange)assert_greater_equal/range/start:output:0"assert_greater_equal/Rank:output:0)assert_greater_equal/range/delta:output:0*
_output_shapes
: 2
assert_greater_equal/range?
assert_greater_equal/AllAll%assert_greater_equal/GreaterEqual:z:0#assert_greater_equal/range:output:0*
_output_shapes
: 2
assert_greater_equal/All?
!assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*?
value?B? B?Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2#
!assert_greater_equal/Assert/Const?
#assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2%
#assert_greater_equal/Assert/Const_1?
#assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2%
#assert_greater_equal/Assert/Const_2?
#assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2%
#assert_greater_equal/Assert/Const_3?
'assert_greater_equal/Assert/AssertGuardIf!assert_greater_equal/All:output:0!assert_greater_equal/All:output:0+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *G
else_branch8R6
4assert_greater_equal_Assert_AssertGuard_false_247562*
output_shapes
: *F
then_branch7R5
3assert_greater_equal_Assert_AssertGuard_true_2475612)
'assert_greater_equal/Assert/AssertGuard?
0assert_greater_equal/Assert/AssertGuard/IdentityIdentity0assert_greater_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identity?
GreaterEqual/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOp?
GreaterEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y?
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqual?
LessEqual/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp?
LessEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqual?
Less/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB :
?????????2
Less/x?
Less/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd?
Sub/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOp?
Sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
Sub?

FloorMod/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorMod?
Equal/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1?
ConstConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Const?
condIfLogicalAnd_1:z:0,assert_greater_equal_readvariableop_resourcecond_input_1cond_input_2cond_input_3LogicalAnd_1:z:0*
Tcond0
*
Tin	
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: *$
_read_only_resource_inputs
*$
else_branchR
cond_false_247602*
output_shapes
: *#
then_branchR
cond_true_2476012
condZ
cond/IdentityIdentitycond:output:0*
T0
*
_output_shapes
: 2
cond/Identityq
updateNoOp1^assert_greater_equal/Assert/AssertGuard/Identity^cond/Identity*
_output_shapes
 2
update}
Mul/ReadVariableOpReadVariableOpcond_input_1*&
_output_shapes
:
*
dtype02
Mul/ReadVariableOp?
Mul/ReadVariableOp_1ReadVariableOpcond_input_2^cond*&
_output_shapes
:
*
dtype02
Mul/ReadVariableOp_1|
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*&
_output_shapes
:
2
Mul?
AssignVariableOpAssignVariableOpcond_input_1Mul:z:0^Mul/ReadVariableOp^cond*
_output_shapes
 *
dtype02
AssignVariableOpw

group_depsNoOp^AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2

group_depsu
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2
group_deps_1?
Conv2D/ReadVariableOpReadVariableOpcond_input_1^AssignVariableOp*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
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
BiasAddc
Relu6Relu6BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu6?
IdentityIdentityRelu6:activations:0^AssignVariableOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp^GreaterEqual/ReadVariableOp^LessEqual/ReadVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1^Sub/ReadVariableOp(^assert_greater_equal/Assert/AssertGuard$^assert_greater_equal/ReadVariableOp^cond*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????(1: : : : : 2$
AssignVariableOpAssignVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2:
GreaterEqual/ReadVariableOpGreaterEqual/ReadVariableOp24
LessEqual/ReadVariableOpLessEqual/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_12(
Sub/ReadVariableOpSub/ReadVariableOp2R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard2J
#assert_greater_equal/ReadVariableOp#assert_greater_equal/ReadVariableOp2
condcond:W S
/
_output_shapes
:?????????(1
 
_user_specified_nameinputs
?
?
U__inference_prune_low_magnitude_dense_layer_call_and_return_conditional_losses_247917

inputs.
mul_readvariableop_resource:	?0
mul_readvariableop_1_resource:	?-
biasadd_readvariableop_resource:
identity??AssignVariableOp?BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?Mul/ReadVariableOp?Mul/ReadVariableOp_14
	no_updateNoOp*
_output_shapes
 2
	no_update?
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
Mul/ReadVariableOp?
Mul/ReadVariableOp_1ReadVariableOpmul_readvariableop_1_resource*
_output_shapes
:	?*
dtype02
Mul/ReadVariableOp_1u
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	?2
Mul?
AssignVariableOpAssignVariableOpmul_readvariableop_resourceMul:z:0^Mul/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpw

group_depsNoOp^AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2

group_depsu
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2
group_deps_1?
MatMul/ReadVariableOpReadVariableOpmul_readvariableop_resource^AssignVariableOp*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
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
IdentityIdentityBiasAdd:output:0^AssignVariableOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:??????????: : : 2$
AssignVariableOpAssignVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_1:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?	
?
6__inference_ExtAudioDataModelPrun_layer_call_fn_246976

inputs
unknown:
	unknown_0:#
	unknown_1:
#
	unknown_2:

	unknown_3:
	unknown_4:	?
	unknown_5:	?
	unknown_6:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_ExtAudioDataModelPrun_layer_call_and_return_conditional_losses_2461932
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????(1: : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????(1
 
_user_specified_nameinputs
?
?
6__inference_ExtAudioDataModelPrun_layer_call_fn_247009

inputs
unknown:
	unknown_0:
	unknown_1:	 #
	unknown_2:
#
	unknown_3:

	unknown_4: 
	unknown_5:
	unknown_6:	 
	unknown_7:	 
	unknown_8:	 
	unknown_9:	?

unknown_10:	?

unknown_11: 

unknown_12:
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
:?????????**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_ExtAudioDataModelPrun_layer_call_and_return_conditional_losses_2467822
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
? 
?
Pprune_low_magnitude_flatten_assert_greater_equal_Assert_AssertGuard_false_247280?
prune_low_magnitude_flatten_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_flatten_assert_greater_equal_all
?
?prune_low_magnitude_flatten_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_flatten_assert_greater_equal_readvariableop	?
}prune_low_magnitude_flatten_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_flatten_assert_greater_equal_y	R
Nprune_low_magnitude_flatten_assert_greater_equal_assert_assertguard_identity_1
??Jprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Assert?
Qprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*?
value?B? B?Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2S
Qprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Assert/data_0?
Qprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2S
Qprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Assert/data_1?
Qprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*Y
valuePBN BHx (prune_low_magnitude_flatten/assert_greater_equal/ReadVariableOp:0) = 2S
Qprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Assert/data_2?
Qprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*L
valueCBA B;y (prune_low_magnitude_flatten/assert_greater_equal/y:0) = 2S
Qprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Assert/data_4?
Jprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/AssertAssertprune_low_magnitude_flatten_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_flatten_assert_greater_equal_allZprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Assert/data_0:output:0Zprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Assert/data_1:output:0Zprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Assert/data_2:output:0?prune_low_magnitude_flatten_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_flatten_assert_greater_equal_readvariableopZprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Assert/data_4:output:0}prune_low_magnitude_flatten_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_flatten_assert_greater_equal_y*
T

2		*
_output_shapes
 2L
Jprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Assert?
Lprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/IdentityIdentityprune_low_magnitude_flatten_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_flatten_assert_greater_equal_allK^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2N
Lprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity?
Nprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity_1IdentityUprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity:output:0K^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2P
Nprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity_1"?
Nprune_low_magnitude_flatten_assert_greater_equal_assert_assertguard_identity_1Wprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2?
Jprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/AssertJprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?.
?
Q__inference_ExtAudioDataModelPrun_layer_call_and_return_conditional_losses_246193

inputs;
-normalization_reshape_readvariableop_resource:=
/normalization_reshape_1_readvariableop_resource:;
!prune_low_magnitude_conv2d_246141:
;
!prune_low_magnitude_conv2d_246143:
/
!prune_low_magnitude_conv2d_246145:3
 prune_low_magnitude_dense_246185:	?3
 prune_low_magnitude_dense_246187:	?.
 prune_low_magnitude_dense_246189:
identity??$normalization/Reshape/ReadVariableOp?&normalization/Reshape_1/ReadVariableOp?2prune_low_magnitude_conv2d/StatefulPartitionedCall?1prune_low_magnitude_dense/StatefulPartitionedCall?
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp?
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape/shape?
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:2
normalization/Reshape?
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp?
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape_1/shape?
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
normalization/Reshape_1?
normalization/subSubinputsnormalization/Reshape:output:0*
T0*/
_output_shapes
:?????????(12
normalization/sub?
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*&
_output_shapes
:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*&
_output_shapes
:2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*/
_output_shapes
:?????????(12
normalization/truediv?
2prune_low_magnitude_conv2d/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0!prune_low_magnitude_conv2d_246141!prune_low_magnitude_conv2d_246143!prune_low_magnitude_conv2d_246145*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_prune_low_magnitude_conv2d_layer_call_and_return_conditional_losses_24614024
2prune_low_magnitude_conv2d/StatefulPartitionedCall?
1prune_low_magnitude_max_pooling2d/PartitionedCallPartitionedCall;prune_low_magnitude_conv2d/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *f
faR_
]__inference_prune_low_magnitude_max_pooling2d_layer_call_and_return_conditional_losses_24615323
1prune_low_magnitude_max_pooling2d/PartitionedCall?
dropout/PartitionedCallPartitionedCall:prune_low_magnitude_max_pooling2d/PartitionedCall:output:0*
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
GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_2461602
dropout/PartitionedCall?
+prune_low_magnitude_flatten/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
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
GPU 2J 8? *`
f[RY
W__inference_prune_low_magnitude_flatten_layer_call_and_return_conditional_losses_2461682-
+prune_low_magnitude_flatten/PartitionedCall?
1prune_low_magnitude_dense/StatefulPartitionedCallStatefulPartitionedCall4prune_low_magnitude_flatten/PartitionedCall:output:0 prune_low_magnitude_dense_246185 prune_low_magnitude_dense_246187 prune_low_magnitude_dense_246189*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_prune_low_magnitude_dense_layer_call_and_return_conditional_losses_24618423
1prune_low_magnitude_dense/StatefulPartitionedCall?
IdentityIdentity:prune_low_magnitude_dense/StatefulPartitionedCall:output:0%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp3^prune_low_magnitude_conv2d/StatefulPartitionedCall2^prune_low_magnitude_dense/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????(1: : : : : : : : 2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp2h
2prune_low_magnitude_conv2d/StatefulPartitionedCall2prune_low_magnitude_conv2d/StatefulPartitionedCall2f
1prune_low_magnitude_dense/StatefulPartitionedCall1prune_low_magnitude_dense/StatefulPartitionedCall:W S
/
_output_shapes
:?????????(1
 
_user_specified_nameinputs
?
?
U__inference_prune_low_magnitude_dense_layer_call_and_return_conditional_losses_246184

inputs.
mul_readvariableop_resource:	?0
mul_readvariableop_1_resource:	?-
biasadd_readvariableop_resource:
identity??AssignVariableOp?BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?Mul/ReadVariableOp?Mul/ReadVariableOp_14
	no_updateNoOp*
_output_shapes
 2
	no_update?
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
Mul/ReadVariableOp?
Mul/ReadVariableOp_1ReadVariableOpmul_readvariableop_1_resource*
_output_shapes
:	?*
dtype02
Mul/ReadVariableOp_1u
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*
_output_shapes
:	?2
Mul?
AssignVariableOpAssignVariableOpmul_readvariableop_resourceMul:z:0^Mul/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpw

group_depsNoOp^AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2

group_depsu
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2
group_deps_1?
MatMul/ReadVariableOpReadVariableOpmul_readvariableop_resource^AssignVariableOp*
_output_shapes
:	?*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
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
IdentityIdentityBiasAdd:output:0^AssignVariableOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*-
_input_shapes
:??????????: : : 2$
AssignVariableOpAssignVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_1:P L
(
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
V__inference_prune_low_magnitude_conv2d_layer_call_and_return_conditional_losses_246140

inputs5
mul_readvariableop_resource:
7
mul_readvariableop_1_resource:
-
biasadd_readvariableop_resource:
identity??AssignVariableOp?BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?Mul/ReadVariableOp?Mul/ReadVariableOp_14
	no_updateNoOp*
_output_shapes
 2
	no_update?
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*&
_output_shapes
:
*
dtype02
Mul/ReadVariableOp?
Mul/ReadVariableOp_1ReadVariableOpmul_readvariableop_1_resource*&
_output_shapes
:
*
dtype02
Mul/ReadVariableOp_1|
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*&
_output_shapes
:
2
Mul?
AssignVariableOpAssignVariableOpmul_readvariableop_resourceMul:z:0^Mul/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpw

group_depsNoOp^AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2

group_depsu
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2
group_deps_1?
Conv2D/ReadVariableOpReadVariableOpmul_readvariableop_resource^AssignVariableOp*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
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
BiasAddc
Relu6Relu6BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu6?
IdentityIdentityRelu6:activations:0^AssignVariableOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????(1: : : 2$
AssignVariableOpAssignVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_1:W S
/
_output_shapes
:?????????(1
 
_user_specified_nameinputs
?.
?
Q__inference_ExtAudioDataModelPrun_layer_call_and_return_conditional_losses_246880
input_1;
-normalization_reshape_readvariableop_resource:=
/normalization_reshape_1_readvariableop_resource:;
!prune_low_magnitude_conv2d_246862:
;
!prune_low_magnitude_conv2d_246864:
/
!prune_low_magnitude_conv2d_246866:3
 prune_low_magnitude_dense_246872:	?3
 prune_low_magnitude_dense_246874:	?.
 prune_low_magnitude_dense_246876:
identity??$normalization/Reshape/ReadVariableOp?&normalization/Reshape_1/ReadVariableOp?2prune_low_magnitude_conv2d/StatefulPartitionedCall?1prune_low_magnitude_dense/StatefulPartitionedCall?
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp?
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape/shape?
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:2
normalization/Reshape?
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp?
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape_1/shape?
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
normalization/Reshape_1?
normalization/subSubinput_1normalization/Reshape:output:0*
T0*/
_output_shapes
:?????????(12
normalization/sub?
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*&
_output_shapes
:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*&
_output_shapes
:2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*/
_output_shapes
:?????????(12
normalization/truediv?
2prune_low_magnitude_conv2d/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0!prune_low_magnitude_conv2d_246862!prune_low_magnitude_conv2d_246864!prune_low_magnitude_conv2d_246866*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_prune_low_magnitude_conv2d_layer_call_and_return_conditional_losses_24614024
2prune_low_magnitude_conv2d/StatefulPartitionedCall?
1prune_low_magnitude_max_pooling2d/PartitionedCallPartitionedCall;prune_low_magnitude_conv2d/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *f
faR_
]__inference_prune_low_magnitude_max_pooling2d_layer_call_and_return_conditional_losses_24615323
1prune_low_magnitude_max_pooling2d/PartitionedCall?
dropout/PartitionedCallPartitionedCall:prune_low_magnitude_max_pooling2d/PartitionedCall:output:0*
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
GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_2461602
dropout/PartitionedCall?
+prune_low_magnitude_flatten/PartitionedCallPartitionedCall dropout/PartitionedCall:output:0*
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
GPU 2J 8? *`
f[RY
W__inference_prune_low_magnitude_flatten_layer_call_and_return_conditional_losses_2461682-
+prune_low_magnitude_flatten/PartitionedCall?
1prune_low_magnitude_dense/StatefulPartitionedCallStatefulPartitionedCall4prune_low_magnitude_flatten/PartitionedCall:output:0 prune_low_magnitude_dense_246872 prune_low_magnitude_dense_246874 prune_low_magnitude_dense_246876*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_prune_low_magnitude_dense_layer_call_and_return_conditional_losses_24618423
1prune_low_magnitude_dense/StatefulPartitionedCall?
IdentityIdentity:prune_low_magnitude_dense/StatefulPartitionedCall:output:0%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp3^prune_low_magnitude_conv2d/StatefulPartitionedCall2^prune_low_magnitude_dense/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*>
_input_shapes-
+:?????????(1: : : : : : : : 2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp2h
2prune_low_magnitude_conv2d/StatefulPartitionedCall2prune_low_magnitude_conv2d/StatefulPartitionedCall2f
1prune_low_magnitude_dense/StatefulPartitionedCall1prune_low_magnitude_dense/StatefulPartitionedCall:X T
/
_output_shapes
:?????????(1
!
_user_specified_name	input_1
?
?
Oprune_low_magnitude_flatten_assert_greater_equal_Assert_AssertGuard_true_247279?
?prune_low_magnitude_flatten_assert_greater_equal_assert_assertguard_identity_prune_low_magnitude_flatten_assert_greater_equal_all
S
Oprune_low_magnitude_flatten_assert_greater_equal_assert_assertguard_placeholder	U
Qprune_low_magnitude_flatten_assert_greater_equal_assert_assertguard_placeholder_1	R
Nprune_low_magnitude_flatten_assert_greater_equal_assert_assertguard_identity_1
?
Hprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2J
Hprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/NoOp?
Lprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/IdentityIdentity?prune_low_magnitude_flatten_assert_greater_equal_assert_assertguard_identity_prune_low_magnitude_flatten_assert_greater_equal_allI^prune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 2N
Lprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity?
Nprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity_1IdentityUprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 2P
Nprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity_1"?
Nprune_low_magnitude_flatten_assert_greater_equal_assert_assertguard_identity_1Wprune_low_magnitude_flatten/assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?7
?
cond_true_2479733
)cond_greaterequal_readvariableop_resource:	 ?
,cond_pruning_ops_abs_readvariableop_resource:	?1
cond_assignvariableop_resource:	?*
 cond_assignvariableop_1_resource: 
cond_identity_logicaland_1

cond_identity_1
??cond/AssignVariableOp?cond/AssignVariableOp_1? cond/GreaterEqual/ReadVariableOp?cond/LessEqual/ReadVariableOp?cond/Sub/ReadVariableOp?#cond/pruning_ops/Abs/ReadVariableOp?
 cond/GreaterEqual/ReadVariableOpReadVariableOp)cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	2"
 cond/GreaterEqual/ReadVariableOpl
cond/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
cond/GreaterEqual/y?
cond/GreaterEqualGreaterEqual(cond/GreaterEqual/ReadVariableOp:value:0cond/GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
cond/GreaterEqual?
cond/LessEqual/ReadVariableOpReadVariableOp)cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	2
cond/LessEqual/ReadVariableOpo
cond/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2
cond/LessEqual/y?
cond/LessEqual	LessEqual%cond/LessEqual/ReadVariableOp:value:0cond/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
cond/LessEquale
cond/Less/xConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
cond/Less/x\
cond/Less/yConst*
_output_shapes
: *
dtype0*
value	B : 2
cond/Less/yk
	cond/LessLesscond/Less/x:output:0cond/Less/y:output:0*
T0*
_output_shapes
: 2
	cond/Lessh
cond/LogicalOr	LogicalOrcond/LessEqual:z:0cond/Less:z:0*
_output_shapes
: 2
cond/LogicalOrs
cond/LogicalAnd
LogicalAndcond/GreaterEqual:z:0cond/LogicalOr:z:0*
_output_shapes
: 2
cond/LogicalAnd?
cond/Sub/ReadVariableOpReadVariableOp)cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	2
cond/Sub/ReadVariableOpZ

cond/Sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2

cond/Sub/yr
cond/SubSubcond/Sub/ReadVariableOp:value:0cond/Sub/y:output:0*
T0	*
_output_shapes
: 2

cond/Subd
cond/FloorMod/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rd2
cond/FloorMod/ys
cond/FloorModFloorModcond/Sub:z:0cond/FloorMod/y:output:0*
T0	*
_output_shapes
: 2
cond/FloorMod^
cond/Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
cond/Equal/yl

cond/EqualEqualcond/FloorMod:z:0cond/Equal/y:output:0*
T0	*
_output_shapes
: 2

cond/Equalq
cond/LogicalAnd_1
LogicalAndcond/LogicalAnd:z:0cond/Equal:z:0*
_output_shapes
: 2
cond/LogicalAnd_1]

cond/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

cond/Const?
#cond/pruning_ops/Abs/ReadVariableOpReadVariableOp,cond_pruning_ops_abs_readvariableop_resource*
_output_shapes
:	?*
dtype02%
#cond/pruning_ops/Abs/ReadVariableOp?
cond/pruning_ops/AbsAbs+cond/pruning_ops/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
cond/pruning_ops/Absq
cond/pruning_ops/SizeConst*
_output_shapes
: *
dtype0*
value
B :?2
cond/pruning_ops/Size?
cond/pruning_ops/CastCastcond/pruning_ops/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
cond/pruning_ops/Castu
cond/pruning_ops/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
cond/pruning_ops/sub/x?
cond/pruning_ops/subSubcond/pruning_ops/sub/x:output:0cond/Const:output:0*
T0*
_output_shapes
: 2
cond/pruning_ops/sub?
cond/pruning_ops/mulMulcond/pruning_ops/Cast:y:0cond/pruning_ops/sub:z:0*
T0*
_output_shapes
: 2
cond/pruning_ops/mult
cond/pruning_ops/RoundRoundcond/pruning_ops/mul:z:0*
T0*
_output_shapes
: 2
cond/pruning_ops/Round?
cond/pruning_ops/Cast_1Castcond/pruning_ops/Round:y:0*

DstT0*

SrcT0*
_output_shapes
: 2
cond/pruning_ops/Cast_1?
cond/pruning_ops/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2 
cond/pruning_ops/Reshape/shape?
cond/pruning_ops/ReshapeReshapecond/pruning_ops/Abs:y:0'cond/pruning_ops/Reshape/shape:output:0*
T0*
_output_shapes	
:?2
cond/pruning_ops/Reshapeu
cond/pruning_ops/Size_1Const*
_output_shapes
: *
dtype0*
value
B :?2
cond/pruning_ops/Size_1?
cond/pruning_ops/TopKV2TopKV2!cond/pruning_ops/Reshape:output:0 cond/pruning_ops/Size_1:output:0*
T0*"
_output_shapes
:?:?2
cond/pruning_ops/TopKV2v
cond/pruning_ops/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
cond/pruning_ops/sub_1/y?
cond/pruning_ops/sub_1Subcond/pruning_ops/Cast_1:y:0!cond/pruning_ops/sub_1/y:output:0*
T0*
_output_shapes
: 2
cond/pruning_ops/sub_1?
cond/pruning_ops/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
cond/pruning_ops/GatherV2/axis?
cond/pruning_ops/GatherV2GatherV2 cond/pruning_ops/TopKV2:values:0cond/pruning_ops/sub_1:z:0'cond/pruning_ops/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: 2
cond/pruning_ops/GatherV2?
cond/pruning_ops/GreaterEqualGreaterEqualcond/pruning_ops/Abs:y:0"cond/pruning_ops/GatherV2:output:0*
T0*
_output_shapes
:	?2
cond/pruning_ops/GreaterEqual?
cond/pruning_ops/Cast_2Cast!cond/pruning_ops/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?2
cond/pruning_ops/Cast_2?
cond/AssignVariableOpAssignVariableOpcond_assignvariableop_resourcecond/pruning_ops/Cast_2:y:0*
_output_shapes
 *
dtype02
cond/AssignVariableOp?
cond/AssignVariableOp_1AssignVariableOp cond_assignvariableop_1_resource"cond/pruning_ops/GatherV2:output:0*
_output_shapes
 *
dtype02
cond/AssignVariableOp_1?
cond/group_depsNoOp^cond/AssignVariableOp^cond/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2
cond/group_depsy
cond/IdentityIdentitycond_identity_logicaland_1^cond/group_deps*
T0
*
_output_shapes
: 2
cond/Identity?
cond/Identity_1Identitycond/Identity:output:0^cond/AssignVariableOp^cond/AssignVariableOp_1!^cond/GreaterEqual/ReadVariableOp^cond/LessEqual/ReadVariableOp^cond/Sub/ReadVariableOp$^cond/pruning_ops/Abs/ReadVariableOp*
T0
*
_output_shapes
: 2
cond/Identity_1"+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2.
cond/AssignVariableOpcond/AssignVariableOp22
cond/AssignVariableOp_1cond/AssignVariableOp_12D
 cond/GreaterEqual/ReadVariableOp cond/GreaterEqual/ReadVariableOp2>
cond/LessEqual/ReadVariableOpcond/LessEqual/ReadVariableOp22
cond/Sub/ReadVariableOpcond/Sub/ReadVariableOp2J
#cond/pruning_ops/Abs/ReadVariableOp#cond/pruning_ops/Abs/ReadVariableOp:

_output_shapes
: 
?
?
Nprune_low_magnitude_dense_assert_greater_equal_Assert_AssertGuard_false_247347
{prune_low_magnitude_dense_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_dense_assert_greater_equal_all
?
?prune_low_magnitude_dense_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_dense_assert_greater_equal_readvariableop	}
yprune_low_magnitude_dense_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_dense_assert_greater_equal_y	P
Lprune_low_magnitude_dense_assert_greater_equal_assert_assertguard_identity_1
??Hprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Assert?
Oprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*?
value?B? B?Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2Q
Oprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Assert/data_0?
Oprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2Q
Oprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Assert/data_1?
Oprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*W
valueNBL BFx (prune_low_magnitude_dense/assert_greater_equal/ReadVariableOp:0) = 2Q
Oprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Assert/data_2?
Oprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*J
valueAB? B9y (prune_low_magnitude_dense/assert_greater_equal/y:0) = 2Q
Oprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Assert/data_4?
Hprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/AssertAssert{prune_low_magnitude_dense_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_dense_assert_greater_equal_allXprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Assert/data_0:output:0Xprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Assert/data_1:output:0Xprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Assert/data_2:output:0?prune_low_magnitude_dense_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_dense_assert_greater_equal_readvariableopXprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Assert/data_4:output:0yprune_low_magnitude_dense_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_dense_assert_greater_equal_y*
T

2		*
_output_shapes
 2J
Hprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Assert?
Jprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/IdentityIdentity{prune_low_magnitude_dense_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_dense_assert_greater_equal_allI^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2L
Jprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity?
Lprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity_1IdentitySprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity:output:0I^prune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2N
Lprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity_1"?
Lprune_low_magnitude_dense_assert_greater_equal_assert_assertguard_identity_1Uprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2?
Hprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/AssertHprune_low_magnitude_dense/assert_greater_equal/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?7
?
cond_true_2462823
)cond_greaterequal_readvariableop_resource:	 ?
,cond_pruning_ops_abs_readvariableop_resource:	?1
cond_assignvariableop_resource:	?*
 cond_assignvariableop_1_resource: 
cond_identity_logicaland_1

cond_identity_1
??cond/AssignVariableOp?cond/AssignVariableOp_1? cond/GreaterEqual/ReadVariableOp?cond/LessEqual/ReadVariableOp?cond/Sub/ReadVariableOp?#cond/pruning_ops/Abs/ReadVariableOp?
 cond/GreaterEqual/ReadVariableOpReadVariableOp)cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	2"
 cond/GreaterEqual/ReadVariableOpl
cond/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
cond/GreaterEqual/y?
cond/GreaterEqualGreaterEqual(cond/GreaterEqual/ReadVariableOp:value:0cond/GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
cond/GreaterEqual?
cond/LessEqual/ReadVariableOpReadVariableOp)cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	2
cond/LessEqual/ReadVariableOpo
cond/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2
cond/LessEqual/y?
cond/LessEqual	LessEqual%cond/LessEqual/ReadVariableOp:value:0cond/LessEqual/y:output:0*
T0	*
_output_shapes
: 2
cond/LessEquale
cond/Less/xConst*
_output_shapes
: *
dtype0*
valueB :
?????????2
cond/Less/x\
cond/Less/yConst*
_output_shapes
: *
dtype0*
value	B : 2
cond/Less/yk
	cond/LessLesscond/Less/x:output:0cond/Less/y:output:0*
T0*
_output_shapes
: 2
	cond/Lessh
cond/LogicalOr	LogicalOrcond/LessEqual:z:0cond/Less:z:0*
_output_shapes
: 2
cond/LogicalOrs
cond/LogicalAnd
LogicalAndcond/GreaterEqual:z:0cond/LogicalOr:z:0*
_output_shapes
: 2
cond/LogicalAnd?
cond/Sub/ReadVariableOpReadVariableOp)cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	2
cond/Sub/ReadVariableOpZ

cond/Sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2

cond/Sub/yr
cond/SubSubcond/Sub/ReadVariableOp:value:0cond/Sub/y:output:0*
T0	*
_output_shapes
: 2

cond/Subd
cond/FloorMod/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rd2
cond/FloorMod/ys
cond/FloorModFloorModcond/Sub:z:0cond/FloorMod/y:output:0*
T0	*
_output_shapes
: 2
cond/FloorMod^
cond/Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
cond/Equal/yl

cond/EqualEqualcond/FloorMod:z:0cond/Equal/y:output:0*
T0	*
_output_shapes
: 2

cond/Equalq
cond/LogicalAnd_1
LogicalAndcond/LogicalAnd:z:0cond/Equal:z:0*
_output_shapes
: 2
cond/LogicalAnd_1]

cond/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2

cond/Const?
#cond/pruning_ops/Abs/ReadVariableOpReadVariableOp,cond_pruning_ops_abs_readvariableop_resource*
_output_shapes
:	?*
dtype02%
#cond/pruning_ops/Abs/ReadVariableOp?
cond/pruning_ops/AbsAbs+cond/pruning_ops/Abs/ReadVariableOp:value:0*
T0*
_output_shapes
:	?2
cond/pruning_ops/Absq
cond/pruning_ops/SizeConst*
_output_shapes
: *
dtype0*
value
B :?2
cond/pruning_ops/Size?
cond/pruning_ops/CastCastcond/pruning_ops/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: 2
cond/pruning_ops/Castu
cond/pruning_ops/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
cond/pruning_ops/sub/x?
cond/pruning_ops/subSubcond/pruning_ops/sub/x:output:0cond/Const:output:0*
T0*
_output_shapes
: 2
cond/pruning_ops/sub?
cond/pruning_ops/mulMulcond/pruning_ops/Cast:y:0cond/pruning_ops/sub:z:0*
T0*
_output_shapes
: 2
cond/pruning_ops/mult
cond/pruning_ops/RoundRoundcond/pruning_ops/mul:z:0*
T0*
_output_shapes
: 2
cond/pruning_ops/Round?
cond/pruning_ops/Cast_1Castcond/pruning_ops/Round:y:0*

DstT0*

SrcT0*
_output_shapes
: 2
cond/pruning_ops/Cast_1?
cond/pruning_ops/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2 
cond/pruning_ops/Reshape/shape?
cond/pruning_ops/ReshapeReshapecond/pruning_ops/Abs:y:0'cond/pruning_ops/Reshape/shape:output:0*
T0*
_output_shapes	
:?2
cond/pruning_ops/Reshapeu
cond/pruning_ops/Size_1Const*
_output_shapes
: *
dtype0*
value
B :?2
cond/pruning_ops/Size_1?
cond/pruning_ops/TopKV2TopKV2!cond/pruning_ops/Reshape:output:0 cond/pruning_ops/Size_1:output:0*
T0*"
_output_shapes
:?:?2
cond/pruning_ops/TopKV2v
cond/pruning_ops/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :2
cond/pruning_ops/sub_1/y?
cond/pruning_ops/sub_1Subcond/pruning_ops/Cast_1:y:0!cond/pruning_ops/sub_1/y:output:0*
T0*
_output_shapes
: 2
cond/pruning_ops/sub_1?
cond/pruning_ops/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2 
cond/pruning_ops/GatherV2/axis?
cond/pruning_ops/GatherV2GatherV2 cond/pruning_ops/TopKV2:values:0cond/pruning_ops/sub_1:z:0'cond/pruning_ops/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: 2
cond/pruning_ops/GatherV2?
cond/pruning_ops/GreaterEqualGreaterEqualcond/pruning_ops/Abs:y:0"cond/pruning_ops/GatherV2:output:0*
T0*
_output_shapes
:	?2
cond/pruning_ops/GreaterEqual?
cond/pruning_ops/Cast_2Cast!cond/pruning_ops/GreaterEqual:z:0*

DstT0*

SrcT0
*
_output_shapes
:	?2
cond/pruning_ops/Cast_2?
cond/AssignVariableOpAssignVariableOpcond_assignvariableop_resourcecond/pruning_ops/Cast_2:y:0*
_output_shapes
 *
dtype02
cond/AssignVariableOp?
cond/AssignVariableOp_1AssignVariableOp cond_assignvariableop_1_resource"cond/pruning_ops/GatherV2:output:0*
_output_shapes
 *
dtype02
cond/AssignVariableOp_1?
cond/group_depsNoOp^cond/AssignVariableOp^cond/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2
cond/group_depsy
cond/IdentityIdentitycond_identity_logicaland_1^cond/group_deps*
T0
*
_output_shapes
: 2
cond/Identity?
cond/Identity_1Identitycond/Identity:output:0^cond/AssignVariableOp^cond/AssignVariableOp_1!^cond/GreaterEqual/ReadVariableOp^cond/LessEqual/ReadVariableOp^cond/Sub/ReadVariableOp$^cond/pruning_ops/Abs/ReadVariableOp*
T0
*
_output_shapes
: 2
cond/Identity_1"+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2.
cond/AssignVariableOpcond/AssignVariableOp22
cond/AssignVariableOp_1cond/AssignVariableOp_12D
 cond/GreaterEqual/ReadVariableOp cond/GreaterEqual/ReadVariableOp2>
cond/LessEqual/ReadVariableOpcond/LessEqual/ReadVariableOp22
cond/Sub/ReadVariableOpcond/Sub/ReadVariableOp2J
#cond/pruning_ops/Abs/ReadVariableOp#cond/pruning_ops/Abs/ReadVariableOp:

_output_shapes
: 
?"
?
Vprune_low_magnitude_max_pooling2d_assert_greater_equal_Assert_AssertGuard_false_247206?
?prune_low_magnitude_max_pooling2d_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_max_pooling2d_assert_greater_equal_all
?
?prune_low_magnitude_max_pooling2d_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_max_pooling2d_assert_greater_equal_readvariableop	?
?prune_low_magnitude_max_pooling2d_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_max_pooling2d_assert_greater_equal_y	X
Tprune_low_magnitude_max_pooling2d_assert_greater_equal_assert_assertguard_identity_1
??Pprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Assert?
Wprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*?
value?B? B?Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2Y
Wprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Assert/data_0?
Wprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2Y
Wprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Assert/data_1?
Wprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*_
valueVBT BNx (prune_low_magnitude_max_pooling2d/assert_greater_equal/ReadVariableOp:0) = 2Y
Wprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Assert/data_2?
Wprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*R
valueIBG BAy (prune_low_magnitude_max_pooling2d/assert_greater_equal/y:0) = 2Y
Wprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Assert/data_4?
Pprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/AssertAssert?prune_low_magnitude_max_pooling2d_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_max_pooling2d_assert_greater_equal_all`prune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Assert/data_0:output:0`prune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Assert/data_1:output:0`prune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Assert/data_2:output:0?prune_low_magnitude_max_pooling2d_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_max_pooling2d_assert_greater_equal_readvariableop`prune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Assert/data_4:output:0?prune_low_magnitude_max_pooling2d_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_max_pooling2d_assert_greater_equal_y*
T

2		*
_output_shapes
 2R
Pprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Assert?
Rprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/IdentityIdentity?prune_low_magnitude_max_pooling2d_assert_greater_equal_assert_assertguard_assert_prune_low_magnitude_max_pooling2d_assert_greater_equal_allQ^prune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2T
Rprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Identity?
Tprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Identity_1Identity[prune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Identity:output:0Q^prune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 2V
Tprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Identity_1"?
Tprune_low_magnitude_max_pooling2d_assert_greater_equal_assert_assertguard_identity_1]prune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2?
Pprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/AssertPprune_low_magnitude_max_pooling2d/assert_greater_equal/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
4assert_greater_equal_Assert_AssertGuard_false_246243K
Gassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all
V
Rassert_greater_equal_assert_assertguard_assert_assert_greater_equal_readvariableop	I
Eassert_greater_equal_assert_assertguard_assert_assert_greater_equal_y	6
2assert_greater_equal_assert_assertguard_identity_1
??.assert_greater_equal/Assert/AssertGuard/Assert?
5assert_greater_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*?
value?B? B?Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.27
5assert_greater_equal/Assert/AssertGuard/Assert/data_0?
5assert_greater_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:27
5assert_greater_equal/Assert/AssertGuard/Assert/data_1?
5assert_greater_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 27
5assert_greater_equal/Assert/AssertGuard/Assert/data_2?
5assert_greater_equal/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 27
5assert_greater_equal/Assert/AssertGuard/Assert/data_4?
.assert_greater_equal/Assert/AssertGuard/AssertAssertGassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all>assert_greater_equal/Assert/AssertGuard/Assert/data_0:output:0>assert_greater_equal/Assert/AssertGuard/Assert/data_1:output:0>assert_greater_equal/Assert/AssertGuard/Assert/data_2:output:0Rassert_greater_equal_assert_assertguard_assert_assert_greater_equal_readvariableop>assert_greater_equal/Assert/AssertGuard/Assert/data_4:output:0Eassert_greater_equal_assert_assertguard_assert_assert_greater_equal_y*
T

2		*
_output_shapes
 20
.assert_greater_equal/Assert/AssertGuard/Assert?
0assert_greater_equal/Assert/AssertGuard/IdentityIdentityGassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all/^assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identity?
2assert_greater_equal/Assert/AssertGuard/Identity_1Identity9assert_greater_equal/Assert/AssertGuard/Identity:output:0/^assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 24
2assert_greater_equal/Assert/AssertGuard/Identity_1"q
2assert_greater_equal_assert_assertguard_identity_1;assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2`
.assert_greater_equal/Assert/AssertGuard/Assert.assert_greater_equal/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
G
cond_true_246538
cond_identity_logicaland_1

cond_identity_1
@
cond/group_depsNoOp*
_output_shapes
 2
cond/group_depsy
cond/IdentityIdentitycond_identity_logicaland_1^cond/group_deps*
T0
*
_output_shapes
: 2
cond/Identityg
cond/Identity_1Identitycond/Identity:output:0*
T0
*
_output_shapes
: 2
cond/Identity_1"+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?7
?	
Q__inference_ExtAudioDataModelPrun_layer_call_and_return_conditional_losses_246782

inputs;
-normalization_reshape_readvariableop_resource:=
/normalization_reshape_1_readvariableop_resource:+
!prune_low_magnitude_conv2d_246752:	 ;
!prune_low_magnitude_conv2d_246754:
;
!prune_low_magnitude_conv2d_246756:
+
!prune_low_magnitude_conv2d_246758: /
!prune_low_magnitude_conv2d_246760:2
(prune_low_magnitude_max_pooling2d_246763:	 ,
"prune_low_magnitude_flatten_246767:	 *
 prune_low_magnitude_dense_246770:	 3
 prune_low_magnitude_dense_246772:	?3
 prune_low_magnitude_dense_246774:	?*
 prune_low_magnitude_dense_246776: .
 prune_low_magnitude_dense_246778:
identity??dropout/StatefulPartitionedCall?$normalization/Reshape/ReadVariableOp?&normalization/Reshape_1/ReadVariableOp?2prune_low_magnitude_conv2d/StatefulPartitionedCall?1prune_low_magnitude_dense/StatefulPartitionedCall?3prune_low_magnitude_flatten/StatefulPartitionedCall?9prune_low_magnitude_max_pooling2d/StatefulPartitionedCall?
$normalization/Reshape/ReadVariableOpReadVariableOp-normalization_reshape_readvariableop_resource*
_output_shapes
:*
dtype02&
$normalization/Reshape/ReadVariableOp?
normalization/Reshape/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape/shape?
normalization/ReshapeReshape,normalization/Reshape/ReadVariableOp:value:0$normalization/Reshape/shape:output:0*
T0*&
_output_shapes
:2
normalization/Reshape?
&normalization/Reshape_1/ReadVariableOpReadVariableOp/normalization_reshape_1_readvariableop_resource*
_output_shapes
:*
dtype02(
&normalization/Reshape_1/ReadVariableOp?
normalization/Reshape_1/shapeConst*
_output_shapes
:*
dtype0*%
valueB"            2
normalization/Reshape_1/shape?
normalization/Reshape_1Reshape.normalization/Reshape_1/ReadVariableOp:value:0&normalization/Reshape_1/shape:output:0*
T0*&
_output_shapes
:2
normalization/Reshape_1?
normalization/subSubinputsnormalization/Reshape:output:0*
T0*/
_output_shapes
:?????????(12
normalization/sub?
normalization/SqrtSqrt normalization/Reshape_1:output:0*
T0*&
_output_shapes
:2
normalization/Sqrtw
normalization/Maximum/yConst*
_output_shapes
: *
dtype0*
valueB
 *???32
normalization/Maximum/y?
normalization/MaximumMaximumnormalization/Sqrt:y:0 normalization/Maximum/y:output:0*
T0*&
_output_shapes
:2
normalization/Maximum?
normalization/truedivRealDivnormalization/sub:z:0normalization/Maximum:z:0*
T0*/
_output_shapes
:?????????(12
normalization/truediv?
2prune_low_magnitude_conv2d/StatefulPartitionedCallStatefulPartitionedCallnormalization/truediv:z:0!prune_low_magnitude_conv2d_246752!prune_low_magnitude_conv2d_246754!prune_low_magnitude_conv2d_246756!prune_low_magnitude_conv2d_246758!prune_low_magnitude_conv2d_246760*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_prune_low_magnitude_conv2d_layer_call_and_return_conditional_losses_24670024
2prune_low_magnitude_conv2d/StatefulPartitionedCall?
9prune_low_magnitude_max_pooling2d/StatefulPartitionedCallStatefulPartitionedCall;prune_low_magnitude_conv2d/StatefulPartitionedCall:output:0(prune_low_magnitude_max_pooling2d_246763*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *f
faR_
]__inference_prune_low_magnitude_max_pooling2d_layer_call_and_return_conditional_losses_2465522;
9prune_low_magnitude_max_pooling2d/StatefulPartitionedCall?
dropout/StatefulPartitionedCallStatefulPartitionedCallBprune_low_magnitude_max_pooling2d/StatefulPartitionedCall:output:0*
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
GPU 2J 8? *L
fGRE
C__inference_dropout_layer_call_and_return_conditional_losses_2464712!
dropout/StatefulPartitionedCall?
3prune_low_magnitude_flatten/StatefulPartitionedCallStatefulPartitionedCall(dropout/StatefulPartitionedCall:output:0"prune_low_magnitude_flatten_246767*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:??????????*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *`
f[RY
W__inference_prune_low_magnitude_flatten_layer_call_and_return_conditional_losses_24644625
3prune_low_magnitude_flatten/StatefulPartitionedCall?
1prune_low_magnitude_dense/StatefulPartitionedCallStatefulPartitionedCall<prune_low_magnitude_flatten/StatefulPartitionedCall:output:0 prune_low_magnitude_dense_246770 prune_low_magnitude_dense_246772 prune_low_magnitude_dense_246774 prune_low_magnitude_dense_246776 prune_low_magnitude_dense_246778*
Tin

2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *^
fYRW
U__inference_prune_low_magnitude_dense_layer_call_and_return_conditional_losses_24635423
1prune_low_magnitude_dense/StatefulPartitionedCall?
IdentityIdentity:prune_low_magnitude_dense/StatefulPartitionedCall:output:0 ^dropout/StatefulPartitionedCall%^normalization/Reshape/ReadVariableOp'^normalization/Reshape_1/ReadVariableOp3^prune_low_magnitude_conv2d/StatefulPartitionedCall2^prune_low_magnitude_dense/StatefulPartitionedCall4^prune_low_magnitude_flatten/StatefulPartitionedCall:^prune_low_magnitude_max_pooling2d/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*J
_input_shapes9
7:?????????(1: : : : : : : : : : : : : : 2B
dropout/StatefulPartitionedCalldropout/StatefulPartitionedCall2L
$normalization/Reshape/ReadVariableOp$normalization/Reshape/ReadVariableOp2P
&normalization/Reshape_1/ReadVariableOp&normalization/Reshape_1/ReadVariableOp2h
2prune_low_magnitude_conv2d/StatefulPartitionedCall2prune_low_magnitude_conv2d/StatefulPartitionedCall2f
1prune_low_magnitude_dense/StatefulPartitionedCall1prune_low_magnitude_dense/StatefulPartitionedCall2j
3prune_low_magnitude_flatten/StatefulPartitionedCall3prune_low_magnitude_flatten/StatefulPartitionedCall2v
9prune_low_magnitude_max_pooling2d/StatefulPartitionedCall9prune_low_magnitude_max_pooling2d/StatefulPartitionedCall:W S
/
_output_shapes
:?????????(1
 
_user_specified_nameinputs
?
?
4assert_greater_equal_Assert_AssertGuard_false_246588K
Gassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all
V
Rassert_greater_equal_assert_assertguard_assert_assert_greater_equal_readvariableop	I
Eassert_greater_equal_assert_assertguard_assert_assert_greater_equal_y	6
2assert_greater_equal_assert_assertguard_identity_1
??.assert_greater_equal/Assert/AssertGuard/Assert?
5assert_greater_equal/Assert/AssertGuard/Assert/data_0Const*
_output_shapes
: *
dtype0*?
value?B? B?Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.27
5assert_greater_equal/Assert/AssertGuard/Assert/data_0?
5assert_greater_equal/Assert/AssertGuard/Assert/data_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:27
5assert_greater_equal/Assert/AssertGuard/Assert/data_1?
5assert_greater_equal/Assert/AssertGuard/Assert/data_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 27
5assert_greater_equal/Assert/AssertGuard/Assert/data_2?
5assert_greater_equal/Assert/AssertGuard/Assert/data_4Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 27
5assert_greater_equal/Assert/AssertGuard/Assert/data_4?
.assert_greater_equal/Assert/AssertGuard/AssertAssertGassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all>assert_greater_equal/Assert/AssertGuard/Assert/data_0:output:0>assert_greater_equal/Assert/AssertGuard/Assert/data_1:output:0>assert_greater_equal/Assert/AssertGuard/Assert/data_2:output:0Rassert_greater_equal_assert_assertguard_assert_assert_greater_equal_readvariableop>assert_greater_equal/Assert/AssertGuard/Assert/data_4:output:0Eassert_greater_equal_assert_assertguard_assert_assert_greater_equal_y*
T

2		*
_output_shapes
 20
.assert_greater_equal/Assert/AssertGuard/Assert?
0assert_greater_equal/Assert/AssertGuard/IdentityIdentityGassert_greater_equal_assert_assertguard_assert_assert_greater_equal_all/^assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identity?
2assert_greater_equal/Assert/AssertGuard/Identity_1Identity9assert_greater_equal/Assert/AssertGuard/Identity:output:0/^assert_greater_equal/Assert/AssertGuard/Assert*
T0
*
_output_shapes
: 24
2assert_greater_equal/Assert/AssertGuard/Identity_1"q
2assert_greater_equal_assert_assertguard_identity_1;assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : 2`
.assert_greater_equal/Assert/AssertGuard/Assert.assert_greater_equal/Assert/AssertGuard/Assert: 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?9
?
]__inference_prune_low_magnitude_max_pooling2d_layer_call_and_return_conditional_losses_247761

inputs6
,assert_greater_equal_readvariableop_resource:	 
identity??GreaterEqual/ReadVariableOp?LessEqual/ReadVariableOp?Sub/ReadVariableOp?'assert_greater_equal/Assert/AssertGuard?#assert_greater_equal/ReadVariableOp?
#assert_greater_equal/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource*
_output_shapes
: *
dtype0	2%
#assert_greater_equal/ReadVariableOpr
assert_greater_equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2
assert_greater_equal/y?
!assert_greater_equal/GreaterEqualGreaterEqual+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
T0	*
_output_shapes
: 2#
!assert_greater_equal/GreaterEqualx
assert_greater_equal/RankConst*
_output_shapes
: *
dtype0*
value	B : 2
assert_greater_equal/Rank?
 assert_greater_equal/range/startConst*
_output_shapes
: *
dtype0*
value	B : 2"
 assert_greater_equal/range/start?
 assert_greater_equal/range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2"
 assert_greater_equal/range/delta?
assert_greater_equal/rangeRange)assert_greater_equal/range/start:output:0"assert_greater_equal/Rank:output:0)assert_greater_equal/range/delta:output:0*
_output_shapes
: 2
assert_greater_equal/range?
assert_greater_equal/AllAll%assert_greater_equal/GreaterEqual:z:0#assert_greater_equal/range:output:0*
_output_shapes
: 2
assert_greater_equal/All?
!assert_greater_equal/Assert/ConstConst*
_output_shapes
: *
dtype0*?
value?B? B?Prune() wrapper requires the UpdatePruningStep callback to be provided during training. Please add it as a callback to your model.fit call.2#
!assert_greater_equal/Assert/Const?
#assert_greater_equal/Assert/Const_1Const*
_output_shapes
: *
dtype0*<
value3B1 B+Condition x >= y did not hold element-wise:2%
#assert_greater_equal/Assert/Const_1?
#assert_greater_equal/Assert/Const_2Const*
_output_shapes
: *
dtype0*=
value4B2 B,x (assert_greater_equal/ReadVariableOp:0) = 2%
#assert_greater_equal/Assert/Const_2?
#assert_greater_equal/Assert/Const_3Const*
_output_shapes
: *
dtype0*0
value'B% By (assert_greater_equal/y:0) = 2%
#assert_greater_equal/Assert/Const_3?
'assert_greater_equal/Assert/AssertGuardIf!assert_greater_equal/All:output:0!assert_greater_equal/All:output:0+assert_greater_equal/ReadVariableOp:value:0assert_greater_equal/y:output:0*
Tcond0
*
Tin
2
		*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *G
else_branch8R6
4assert_greater_equal_Assert_AssertGuard_false_247708*
output_shapes
: *F
then_branch7R5
3assert_greater_equal_Assert_AssertGuard_true_2477072)
'assert_greater_equal/Assert/AssertGuard?
0assert_greater_equal/Assert/AssertGuard/IdentityIdentity0assert_greater_equal/Assert/AssertGuard:output:0*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identity?
GreaterEqual/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
GreaterEqual/ReadVariableOp?
GreaterEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
GreaterEqual/y?
GreaterEqualGreaterEqual#GreaterEqual/ReadVariableOp:value:0GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2
GreaterEqual?
LessEqual/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
LessEqual/ReadVariableOp?
LessEqual/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2
LessEqual/y|
	LessEqual	LessEqual LessEqual/ReadVariableOp:value:0LessEqual/y:output:0*
T0	*
_output_shapes
: 2
	LessEqual?
Less/xConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB :
?????????2
Less/x?
Less/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
value	B : 2
Less/yW
LessLessLess/x:output:0Less/y:output:0*
T0*
_output_shapes
: 2
LessT
	LogicalOr	LogicalOrLessEqual:z:0Less:z:0*
_output_shapes
: 2
	LogicalOr_

LogicalAnd
LogicalAndGreaterEqual:z:0LogicalOr:z:0*
_output_shapes
: 2

LogicalAnd?
Sub/ReadVariableOpReadVariableOp,assert_greater_equal_readvariableop_resource1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	2
Sub/ReadVariableOp?
Sub/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2
Sub/y^
SubSubSub/ReadVariableOp:value:0Sub/y:output:0*
T0	*
_output_shapes
: 2
Sub?

FloorMod/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 Rd2

FloorMod/y_
FloorModFloorModSub:z:0FloorMod/y:output:0*
T0	*
_output_shapes
: 2

FloorMod?
Equal/yConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0	*
value	B	 R 2	
Equal/yX
EqualEqualFloorMod:z:0Equal/y:output:0*
T0	*
_output_shapes
: 2
Equal]
LogicalAnd_1
LogicalAndLogicalAnd:z:0	Equal:z:0*
_output_shapes
: 2
LogicalAnd_1?
ConstConst1^assert_greater_equal/Assert/AssertGuard/Identity*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Const?
condStatelessIfLogicalAnd_1:z:0LogicalAnd_1:z:0*
Tcond0
*
Tin
2
*
Tout
2
*
_lower_using_switch_merge(*
_output_shapes
: * 
_read_only_resource_inputs
 *$
else_branchR
cond_false_247748*
output_shapes
: *#
then_branchR
cond_true_2477472
condZ
cond/IdentityIdentitycond:output:0*
T0
*
_output_shapes
: 2
cond/Identityq
updateNoOp1^assert_greater_equal/Assert/AssertGuard/Identity^cond/Identity*
_output_shapes
 2
update6

group_depsNoOp*
_output_shapes
 2

group_deps?
MaxPoolMaxPoolinputs*/
_output_shapes
:?????????
*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0^GreaterEqual/ReadVariableOp^LessEqual/ReadVariableOp^Sub/ReadVariableOp(^assert_greater_equal/Assert/AssertGuard$^assert_greater_equal/ReadVariableOp*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: 2:
GreaterEqual/ReadVariableOpGreaterEqual/ReadVariableOp24
LessEqual/ReadVariableOpLessEqual/ReadVariableOp2(
Sub/ReadVariableOpSub/ReadVariableOp2R
'assert_greater_equal/Assert/AssertGuard'assert_greater_equal/Assert/AssertGuard2J
#assert_greater_equal/ReadVariableOp#assert_greater_equal/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
s
W__inference_prune_low_magnitude_flatten_layer_call_and_return_conditional_losses_247806

inputs
identity4
	no_updateNoOp*
_output_shapes
 2
	no_update6

group_depsNoOp*
_output_shapes
 2

group_deps_
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
?\
?
+prune_low_magnitude_conv2d_cond_true_247120N
Dprune_low_magnitude_conv2d_cond_greaterequal_readvariableop_resource:	 a
Gprune_low_magnitude_conv2d_cond_pruning_ops_abs_readvariableop_resource:
S
9prune_low_magnitude_conv2d_cond_assignvariableop_resource:
E
;prune_low_magnitude_conv2d_cond_assignvariableop_1_resource: T
Pprune_low_magnitude_conv2d_cond_identity_prune_low_magnitude_conv2d_logicaland_1
.
*prune_low_magnitude_conv2d_cond_identity_1
??0prune_low_magnitude_conv2d/cond/AssignVariableOp?2prune_low_magnitude_conv2d/cond/AssignVariableOp_1?;prune_low_magnitude_conv2d/cond/GreaterEqual/ReadVariableOp?8prune_low_magnitude_conv2d/cond/LessEqual/ReadVariableOp?2prune_low_magnitude_conv2d/cond/Sub/ReadVariableOp?>prune_low_magnitude_conv2d/cond/pruning_ops/Abs/ReadVariableOp?
;prune_low_magnitude_conv2d/cond/GreaterEqual/ReadVariableOpReadVariableOpDprune_low_magnitude_conv2d_cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	2=
;prune_low_magnitude_conv2d/cond/GreaterEqual/ReadVariableOp?
.prune_low_magnitude_conv2d/cond/GreaterEqual/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 20
.prune_low_magnitude_conv2d/cond/GreaterEqual/y?
,prune_low_magnitude_conv2d/cond/GreaterEqualGreaterEqualCprune_low_magnitude_conv2d/cond/GreaterEqual/ReadVariableOp:value:07prune_low_magnitude_conv2d/cond/GreaterEqual/y:output:0*
T0	*
_output_shapes
: 2.
,prune_low_magnitude_conv2d/cond/GreaterEqual?
8prune_low_magnitude_conv2d/cond/LessEqual/ReadVariableOpReadVariableOpDprune_low_magnitude_conv2d_cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	2:
8prune_low_magnitude_conv2d/cond/LessEqual/ReadVariableOp?
+prune_low_magnitude_conv2d/cond/LessEqual/yConst*
_output_shapes
: *
dtype0	*
valueB	 R
?????????2-
+prune_low_magnitude_conv2d/cond/LessEqual/y?
)prune_low_magnitude_conv2d/cond/LessEqual	LessEqual@prune_low_magnitude_conv2d/cond/LessEqual/ReadVariableOp:value:04prune_low_magnitude_conv2d/cond/LessEqual/y:output:0*
T0	*
_output_shapes
: 2+
)prune_low_magnitude_conv2d/cond/LessEqual?
&prune_low_magnitude_conv2d/cond/Less/xConst*
_output_shapes
: *
dtype0*
valueB :
?????????2(
&prune_low_magnitude_conv2d/cond/Less/x?
&prune_low_magnitude_conv2d/cond/Less/yConst*
_output_shapes
: *
dtype0*
value	B : 2(
&prune_low_magnitude_conv2d/cond/Less/y?
$prune_low_magnitude_conv2d/cond/LessLess/prune_low_magnitude_conv2d/cond/Less/x:output:0/prune_low_magnitude_conv2d/cond/Less/y:output:0*
T0*
_output_shapes
: 2&
$prune_low_magnitude_conv2d/cond/Less?
)prune_low_magnitude_conv2d/cond/LogicalOr	LogicalOr-prune_low_magnitude_conv2d/cond/LessEqual:z:0(prune_low_magnitude_conv2d/cond/Less:z:0*
_output_shapes
: 2+
)prune_low_magnitude_conv2d/cond/LogicalOr?
*prune_low_magnitude_conv2d/cond/LogicalAnd
LogicalAnd0prune_low_magnitude_conv2d/cond/GreaterEqual:z:0-prune_low_magnitude_conv2d/cond/LogicalOr:z:0*
_output_shapes
: 2,
*prune_low_magnitude_conv2d/cond/LogicalAnd?
2prune_low_magnitude_conv2d/cond/Sub/ReadVariableOpReadVariableOpDprune_low_magnitude_conv2d_cond_greaterequal_readvariableop_resource*
_output_shapes
: *
dtype0	24
2prune_low_magnitude_conv2d/cond/Sub/ReadVariableOp?
%prune_low_magnitude_conv2d/cond/Sub/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2'
%prune_low_magnitude_conv2d/cond/Sub/y?
#prune_low_magnitude_conv2d/cond/SubSub:prune_low_magnitude_conv2d/cond/Sub/ReadVariableOp:value:0.prune_low_magnitude_conv2d/cond/Sub/y:output:0*
T0	*
_output_shapes
: 2%
#prune_low_magnitude_conv2d/cond/Sub?
*prune_low_magnitude_conv2d/cond/FloorMod/yConst*
_output_shapes
: *
dtype0	*
value	B	 Rd2,
*prune_low_magnitude_conv2d/cond/FloorMod/y?
(prune_low_magnitude_conv2d/cond/FloorModFloorMod'prune_low_magnitude_conv2d/cond/Sub:z:03prune_low_magnitude_conv2d/cond/FloorMod/y:output:0*
T0	*
_output_shapes
: 2*
(prune_low_magnitude_conv2d/cond/FloorMod?
'prune_low_magnitude_conv2d/cond/Equal/yConst*
_output_shapes
: *
dtype0	*
value	B	 R 2)
'prune_low_magnitude_conv2d/cond/Equal/y?
%prune_low_magnitude_conv2d/cond/EqualEqual,prune_low_magnitude_conv2d/cond/FloorMod:z:00prune_low_magnitude_conv2d/cond/Equal/y:output:0*
T0	*
_output_shapes
: 2'
%prune_low_magnitude_conv2d/cond/Equal?
,prune_low_magnitude_conv2d/cond/LogicalAnd_1
LogicalAnd.prune_low_magnitude_conv2d/cond/LogicalAnd:z:0)prune_low_magnitude_conv2d/cond/Equal:z:0*
_output_shapes
: 2.
,prune_low_magnitude_conv2d/cond/LogicalAnd_1?
%prune_low_magnitude_conv2d/cond/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2'
%prune_low_magnitude_conv2d/cond/Const?
>prune_low_magnitude_conv2d/cond/pruning_ops/Abs/ReadVariableOpReadVariableOpGprune_low_magnitude_conv2d_cond_pruning_ops_abs_readvariableop_resource*&
_output_shapes
:
*
dtype02@
>prune_low_magnitude_conv2d/cond/pruning_ops/Abs/ReadVariableOp?
/prune_low_magnitude_conv2d/cond/pruning_ops/AbsAbsFprune_low_magnitude_conv2d/cond/pruning_ops/Abs/ReadVariableOp:value:0*
T0*&
_output_shapes
:
21
/prune_low_magnitude_conv2d/cond/pruning_ops/Abs?
0prune_low_magnitude_conv2d/cond/pruning_ops/SizeConst*
_output_shapes
: *
dtype0*
value
B :?22
0prune_low_magnitude_conv2d/cond/pruning_ops/Size?
0prune_low_magnitude_conv2d/cond/pruning_ops/CastCast9prune_low_magnitude_conv2d/cond/pruning_ops/Size:output:0*

DstT0*

SrcT0*
_output_shapes
: 22
0prune_low_magnitude_conv2d/cond/pruning_ops/Cast?
1prune_low_magnitude_conv2d/cond/pruning_ops/sub/xConst*
_output_shapes
: *
dtype0*
valueB
 *  ??23
1prune_low_magnitude_conv2d/cond/pruning_ops/sub/x?
/prune_low_magnitude_conv2d/cond/pruning_ops/subSub:prune_low_magnitude_conv2d/cond/pruning_ops/sub/x:output:0.prune_low_magnitude_conv2d/cond/Const:output:0*
T0*
_output_shapes
: 21
/prune_low_magnitude_conv2d/cond/pruning_ops/sub?
/prune_low_magnitude_conv2d/cond/pruning_ops/mulMul4prune_low_magnitude_conv2d/cond/pruning_ops/Cast:y:03prune_low_magnitude_conv2d/cond/pruning_ops/sub:z:0*
T0*
_output_shapes
: 21
/prune_low_magnitude_conv2d/cond/pruning_ops/mul?
1prune_low_magnitude_conv2d/cond/pruning_ops/RoundRound3prune_low_magnitude_conv2d/cond/pruning_ops/mul:z:0*
T0*
_output_shapes
: 23
1prune_low_magnitude_conv2d/cond/pruning_ops/Round?
2prune_low_magnitude_conv2d/cond/pruning_ops/Cast_1Cast5prune_low_magnitude_conv2d/cond/pruning_ops/Round:y:0*

DstT0*

SrcT0*
_output_shapes
: 24
2prune_low_magnitude_conv2d/cond/pruning_ops/Cast_1?
9prune_low_magnitude_conv2d/cond/pruning_ops/Reshape/shapeConst*
_output_shapes
:*
dtype0*
valueB:
?????????2;
9prune_low_magnitude_conv2d/cond/pruning_ops/Reshape/shape?
3prune_low_magnitude_conv2d/cond/pruning_ops/ReshapeReshape3prune_low_magnitude_conv2d/cond/pruning_ops/Abs:y:0Bprune_low_magnitude_conv2d/cond/pruning_ops/Reshape/shape:output:0*
T0*
_output_shapes	
:?25
3prune_low_magnitude_conv2d/cond/pruning_ops/Reshape?
2prune_low_magnitude_conv2d/cond/pruning_ops/Size_1Const*
_output_shapes
: *
dtype0*
value
B :?24
2prune_low_magnitude_conv2d/cond/pruning_ops/Size_1?
2prune_low_magnitude_conv2d/cond/pruning_ops/TopKV2TopKV2<prune_low_magnitude_conv2d/cond/pruning_ops/Reshape:output:0;prune_low_magnitude_conv2d/cond/pruning_ops/Size_1:output:0*
T0*"
_output_shapes
:?:?24
2prune_low_magnitude_conv2d/cond/pruning_ops/TopKV2?
3prune_low_magnitude_conv2d/cond/pruning_ops/sub_1/yConst*
_output_shapes
: *
dtype0*
value	B :25
3prune_low_magnitude_conv2d/cond/pruning_ops/sub_1/y?
1prune_low_magnitude_conv2d/cond/pruning_ops/sub_1Sub6prune_low_magnitude_conv2d/cond/pruning_ops/Cast_1:y:0<prune_low_magnitude_conv2d/cond/pruning_ops/sub_1/y:output:0*
T0*
_output_shapes
: 23
1prune_low_magnitude_conv2d/cond/pruning_ops/sub_1?
9prune_low_magnitude_conv2d/cond/pruning_ops/GatherV2/axisConst*
_output_shapes
: *
dtype0*
value	B : 2;
9prune_low_magnitude_conv2d/cond/pruning_ops/GatherV2/axis?
4prune_low_magnitude_conv2d/cond/pruning_ops/GatherV2GatherV2;prune_low_magnitude_conv2d/cond/pruning_ops/TopKV2:values:05prune_low_magnitude_conv2d/cond/pruning_ops/sub_1:z:0Bprune_low_magnitude_conv2d/cond/pruning_ops/GatherV2/axis:output:0*
Taxis0*
Tindices0*
Tparams0*
_output_shapes
: 26
4prune_low_magnitude_conv2d/cond/pruning_ops/GatherV2?
8prune_low_magnitude_conv2d/cond/pruning_ops/GreaterEqualGreaterEqual3prune_low_magnitude_conv2d/cond/pruning_ops/Abs:y:0=prune_low_magnitude_conv2d/cond/pruning_ops/GatherV2:output:0*
T0*&
_output_shapes
:
2:
8prune_low_magnitude_conv2d/cond/pruning_ops/GreaterEqual?
2prune_low_magnitude_conv2d/cond/pruning_ops/Cast_2Cast<prune_low_magnitude_conv2d/cond/pruning_ops/GreaterEqual:z:0*

DstT0*

SrcT0
*&
_output_shapes
:
24
2prune_low_magnitude_conv2d/cond/pruning_ops/Cast_2?
0prune_low_magnitude_conv2d/cond/AssignVariableOpAssignVariableOp9prune_low_magnitude_conv2d_cond_assignvariableop_resource6prune_low_magnitude_conv2d/cond/pruning_ops/Cast_2:y:0*
_output_shapes
 *
dtype022
0prune_low_magnitude_conv2d/cond/AssignVariableOp?
2prune_low_magnitude_conv2d/cond/AssignVariableOp_1AssignVariableOp;prune_low_magnitude_conv2d_cond_assignvariableop_1_resource=prune_low_magnitude_conv2d/cond/pruning_ops/GatherV2:output:0*
_output_shapes
 *
dtype024
2prune_low_magnitude_conv2d/cond/AssignVariableOp_1?
*prune_low_magnitude_conv2d/cond/group_depsNoOp1^prune_low_magnitude_conv2d/cond/AssignVariableOp3^prune_low_magnitude_conv2d/cond/AssignVariableOp_1",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2,
*prune_low_magnitude_conv2d/cond/group_deps?
(prune_low_magnitude_conv2d/cond/IdentityIdentityPprune_low_magnitude_conv2d_cond_identity_prune_low_magnitude_conv2d_logicaland_1+^prune_low_magnitude_conv2d/cond/group_deps*
T0
*
_output_shapes
: 2*
(prune_low_magnitude_conv2d/cond/Identity?
*prune_low_magnitude_conv2d/cond/Identity_1Identity1prune_low_magnitude_conv2d/cond/Identity:output:01^prune_low_magnitude_conv2d/cond/AssignVariableOp3^prune_low_magnitude_conv2d/cond/AssignVariableOp_1<^prune_low_magnitude_conv2d/cond/GreaterEqual/ReadVariableOp9^prune_low_magnitude_conv2d/cond/LessEqual/ReadVariableOp3^prune_low_magnitude_conv2d/cond/Sub/ReadVariableOp?^prune_low_magnitude_conv2d/cond/pruning_ops/Abs/ReadVariableOp*
T0
*
_output_shapes
: 2,
*prune_low_magnitude_conv2d/cond/Identity_1"a
*prune_low_magnitude_conv2d_cond_identity_13prune_low_magnitude_conv2d/cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : 2d
0prune_low_magnitude_conv2d/cond/AssignVariableOp0prune_low_magnitude_conv2d/cond/AssignVariableOp2h
2prune_low_magnitude_conv2d/cond/AssignVariableOp_12prune_low_magnitude_conv2d/cond/AssignVariableOp_12z
;prune_low_magnitude_conv2d/cond/GreaterEqual/ReadVariableOp;prune_low_magnitude_conv2d/cond/GreaterEqual/ReadVariableOp2t
8prune_low_magnitude_conv2d/cond/LessEqual/ReadVariableOp8prune_low_magnitude_conv2d/cond/LessEqual/ReadVariableOp2h
2prune_low_magnitude_conv2d/cond/Sub/ReadVariableOp2prune_low_magnitude_conv2d/cond/Sub/ReadVariableOp2?
>prune_low_magnitude_conv2d/cond/pruning_ops/Abs/ReadVariableOp>prune_low_magnitude_conv2d/cond/pruning_ops/Abs/ReadVariableOp:

_output_shapes
: 
?
^
B__inference_prune_low_magnitude_max_pooling2d_layer_call_fn_247679

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
GPU 2J 8? *f
faR_
]__inference_prune_low_magnitude_max_pooling2d_layer_call_and_return_conditional_losses_2461532
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
?
b
C__inference_dropout_layer_call_and_return_conditional_losses_246471

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
?
?
B__inference_prune_low_magnitude_max_pooling2d_layer_call_fn_247686

inputs
unknown:	 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????
*#
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *f
faR_
]__inference_prune_low_magnitude_max_pooling2d_layer_call_and_return_conditional_losses_2465522
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*0
_input_shapes
:?????????: 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
;__inference_prune_low_magnitude_conv2d_layer_call_fn_247530

inputs
unknown:	 #
	unknown_0:
#
	unknown_1:

	unknown_2: 
	unknown_3:
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3*
Tin

2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *_
fZRX
V__inference_prune_low_magnitude_conv2d_layer_call_and_return_conditional_losses_2467002
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*8
_input_shapes'
%:?????????(1: : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????(1
 
_user_specified_nameinputs
?
?
V__inference_prune_low_magnitude_conv2d_layer_call_and_return_conditional_losses_247545

inputs5
mul_readvariableop_resource:
7
mul_readvariableop_1_resource:
-
biasadd_readvariableop_resource:
identity??AssignVariableOp?BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?Mul/ReadVariableOp?Mul/ReadVariableOp_14
	no_updateNoOp*
_output_shapes
 2
	no_update?
Mul/ReadVariableOpReadVariableOpmul_readvariableop_resource*&
_output_shapes
:
*
dtype02
Mul/ReadVariableOp?
Mul/ReadVariableOp_1ReadVariableOpmul_readvariableop_1_resource*&
_output_shapes
:
*
dtype02
Mul/ReadVariableOp_1|
MulMulMul/ReadVariableOp:value:0Mul/ReadVariableOp_1:value:0*
T0*&
_output_shapes
:
2
Mul?
AssignVariableOpAssignVariableOpmul_readvariableop_resourceMul:z:0^Mul/ReadVariableOp*
_output_shapes
 *
dtype02
AssignVariableOpw

group_depsNoOp^AssignVariableOp",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2

group_depsu
group_deps_1NoOp^group_deps",/job:localhost/replica:0/task:0/device:CPU:0*
_output_shapes
 2
group_deps_1?
Conv2D/ReadVariableOpReadVariableOpmul_readvariableop_resource^AssignVariableOp*&
_output_shapes
:
*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
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
BiasAddc
Relu6Relu6BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
Relu6?
IdentityIdentityRelu6:activations:0^AssignVariableOp^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp^Mul/ReadVariableOp^Mul/ReadVariableOp_1*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*4
_input_shapes#
!:?????????(1: : : 2$
AssignVariableOpAssignVariableOp20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp2(
Mul/ReadVariableOpMul/ReadVariableOp2,
Mul/ReadVariableOp_1Mul/ReadVariableOp_1:W S
/
_output_shapes
:?????????(1
 
_user_specified_nameinputs
?
G
cond_true_247747
cond_identity_logicaland_1

cond_identity_1
@
cond/group_depsNoOp*
_output_shapes
 2
cond/group_depsy
cond/IdentityIdentitycond_identity_logicaland_1^cond/group_deps*
T0
*
_output_shapes
: 2
cond/Identityg
cond/Identity_1Identitycond/Identity:output:0*
T0
*
_output_shapes
: 2
cond/Identity_1"+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_247776

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
?
H
cond_false_246432
cond_identity_logicaland_1

cond_identity_1
4
	cond/NoOpNoOp*
_output_shapes
 2
	cond/NoOps
cond/IdentityIdentitycond_identity_logicaland_1
^cond/NoOp*
T0
*
_output_shapes
: 2
cond/Identityg
cond/Identity_1Identitycond/Identity:output:0*
T0
*
_output_shapes
: 2
cond/Identity_1"+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
G
cond_true_247862
cond_identity_logicaland_1

cond_identity_1
@
cond/group_depsNoOp*
_output_shapes
 2
cond/group_depsy
cond/IdentityIdentitycond_identity_logicaland_1^cond/group_deps*
T0
*
_output_shapes
: 2
cond/Identityg
cond/Identity_1Identitycond/Identity:output:0*
T0
*
_output_shapes
: 2
cond/Identity_1"+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?	
?
3assert_greater_equal_Assert_AssertGuard_true_247707M
Iassert_greater_equal_assert_assertguard_identity_assert_greater_equal_all
7
3assert_greater_equal_assert_assertguard_placeholder	9
5assert_greater_equal_assert_assertguard_placeholder_1	6
2assert_greater_equal_assert_assertguard_identity_1
z
,assert_greater_equal/Assert/AssertGuard/NoOpNoOp*
_output_shapes
 2.
,assert_greater_equal/Assert/AssertGuard/NoOp?
0assert_greater_equal/Assert/AssertGuard/IdentityIdentityIassert_greater_equal_assert_assertguard_identity_assert_greater_equal_all-^assert_greater_equal/Assert/AssertGuard/NoOp*
T0
*
_output_shapes
: 22
0assert_greater_equal/Assert/AssertGuard/Identity?
2assert_greater_equal/Assert/AssertGuard/Identity_1Identity9assert_greater_equal/Assert/AssertGuard/Identity:output:0*
T0
*
_output_shapes
: 24
2assert_greater_equal/Assert/AssertGuard/Identity_1"q
2assert_greater_equal_assert_assertguard_identity_1;assert_greater_equal/Assert/AssertGuard/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : : : 

_output_shapes
: :

_output_shapes
: :

_output_shapes
: 
?
?
cond_false_247602
cond_placeholder
cond_placeholder_1
cond_placeholder_2
cond_placeholder_3
cond_identity_logicaland_1

cond_identity_1
4
	cond/NoOpNoOp*
_output_shapes
 2
	cond/NoOps
cond/IdentityIdentitycond_identity_logicaland_1
^cond/NoOp*
T0
*
_output_shapes
: 2
cond/Identityg
cond/Identity_1Identitycond/Identity:output:0*
T0
*
_output_shapes
: 2
cond/Identity_1"+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes

: : : : : :

_output_shapes
: 
?
H
cond_false_247748
cond_identity_logicaland_1

cond_identity_1
4
	cond/NoOpNoOp*
_output_shapes
 2
	cond/NoOps
cond/IdentityIdentitycond_identity_logicaland_1
^cond/NoOp*
T0
*
_output_shapes
: 2
cond/Identityg
cond/Identity_1Identitycond/Identity:output:0*
T0
*
_output_shapes
: 2
cond/Identity_1"+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?D
?
__inference__traced_save_248164
file_prefix#
savev2_mean_read_readvariableop'
#savev2_variance_read_readvariableop$
 savev2_count_read_readvariableop	>
:savev2_prune_low_magnitude_conv2d_mask_read_readvariableopC
?savev2_prune_low_magnitude_conv2d_threshold_read_readvariableopF
Bsavev2_prune_low_magnitude_conv2d_pruning_step_read_readvariableop	M
Isavev2_prune_low_magnitude_max_pooling2d_pruning_step_read_readvariableop	G
Csavev2_prune_low_magnitude_flatten_pruning_step_read_readvariableop	=
9savev2_prune_low_magnitude_dense_mask_read_readvariableopB
>savev2_prune_low_magnitude_dense_threshold_read_readvariableopE
Asavev2_prune_low_magnitude_dense_pruning_step_read_readvariableop	#
savev2_iter_read_readvariableop	%
!savev2_beta_1_read_readvariableop%
!savev2_beta_2_read_readvariableop$
 savev2_decay_read_readvariableop,
(savev2_learning_rate_read_readvariableop,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop$
 savev2_total_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_2_read_readvariableop3
/savev2_adam_conv2d_kernel_m_read_readvariableop1
-savev2_adam_conv2d_bias_m_read_readvariableop2
.savev2_adam_dense_kernel_m_read_readvariableop0
,savev2_adam_dense_bias_m_read_readvariableop3
/savev2_adam_conv2d_kernel_v_read_readvariableop1
-savev2_adam_conv2d_bias_v_read_readvariableop2
.savev2_adam_dense_kernel_v_read_readvariableop0
,savev2_adam_dense_bias_v_read_readvariableop
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
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*?
value?B?!B4layer_with_weights-0/mean/.ATTRIBUTES/VARIABLE_VALUEB8layer_with_weights-0/variance/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-0/count/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/mask/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-1/threshold/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-1/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-2/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-3/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/mask/.ATTRIBUTES/VARIABLE_VALUEB9layer_with_weights-4/threshold/.ATTRIBUTES/VARIABLE_VALUEB<layer_with_weights-4/pruning_step/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:!*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0savev2_mean_read_readvariableop#savev2_variance_read_readvariableop savev2_count_read_readvariableop:savev2_prune_low_magnitude_conv2d_mask_read_readvariableop?savev2_prune_low_magnitude_conv2d_threshold_read_readvariableopBsavev2_prune_low_magnitude_conv2d_pruning_step_read_readvariableopIsavev2_prune_low_magnitude_max_pooling2d_pruning_step_read_readvariableopCsavev2_prune_low_magnitude_flatten_pruning_step_read_readvariableop9savev2_prune_low_magnitude_dense_mask_read_readvariableop>savev2_prune_low_magnitude_dense_threshold_read_readvariableopAsavev2_prune_low_magnitude_dense_pruning_step_read_readvariableopsavev2_iter_read_readvariableop!savev2_beta_1_read_readvariableop!savev2_beta_2_read_readvariableop savev2_decay_read_readvariableop(savev2_learning_rate_read_readvariableop(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop savev2_total_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_2_read_readvariableop/savev2_adam_conv2d_kernel_m_read_readvariableop-savev2_adam_conv2d_bias_m_read_readvariableop.savev2_adam_dense_kernel_m_read_readvariableop,savev2_adam_dense_bias_m_read_readvariableop/savev2_adam_conv2d_kernel_v_read_readvariableop-savev2_adam_conv2d_bias_v_read_readvariableop.savev2_adam_dense_kernel_v_read_readvariableop,savev2_adam_dense_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 */
dtypes%
#2!						2
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
?: ::: :
: : : : :	?: : : : : : : :
::	?:: : : : :
::	?::
::	?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix: 

_output_shapes
:: 

_output_shapes
::

_output_shapes
: :,(
&
_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :%	!

_output_shapes
:	?:
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
: :,(
&
_output_shapes
:
: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
:
: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::,(
&
_output_shapes
:
: 

_output_shapes
::%!

_output_shapes
:	?:  

_output_shapes
::!

_output_shapes
: 
?
?
6__inference_ExtAudioDataModelPrun_layer_call_fn_246846
input_1
unknown:
	unknown_0:
	unknown_1:	 #
	unknown_2:
#
	unknown_3:

	unknown_4: 
	unknown_5:
	unknown_6:	 
	unknown_7:	 
	unknown_8:	 
	unknown_9:	?

unknown_10:	?

unknown_11: 

unknown_12:
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
:?????????**
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *Z
fURS
Q__inference_ExtAudioDataModelPrun_layer_call_and_return_conditional_losses_2467822
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
?
H
cond_false_247863
cond_identity_logicaland_1

cond_identity_1
4
	cond/NoOpNoOp*
_output_shapes
 2
	cond/NoOps
cond/IdentityIdentitycond_identity_logicaland_1
^cond/NoOp*
T0
*
_output_shapes
: 2
cond/Identityg
cond/Identity_1Identitycond/Identity:output:0*
T0
*
_output_shapes
: 2
cond/Identity_1"+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: 
?
a
C__inference_dropout_layer_call_and_return_conditional_losses_246160

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
?
H
cond_false_246539
cond_identity_logicaland_1

cond_identity_1
4
	cond/NoOpNoOp*
_output_shapes
 2
	cond/NoOps
cond/IdentityIdentitycond_identity_logicaland_1
^cond/NoOp*
T0
*
_output_shapes
: 2
cond/Identityg
cond/Identity_1Identitycond/Identity:output:0*
T0
*
_output_shapes
: 2
cond/Identity_1"+
cond_identity_1cond/Identity_1:output:0*(
_construction_contextkEagerRuntime*
_input_shapes
: : 

_output_shapes
: "?L
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
serving_default_input_1:0?????????(1M
prune_low_magnitude_dense0
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?H
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
	optimizer
trainable_variables
	regularization_losses

	variables
	keras_api

signatures
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses"?E
_tf_keras_sequential?D{"name": "ExtAudioDataModelPrun", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Sequential", "config": {"name": "ExtAudioDataModelPrun", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 49, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}, {"class_name": "Normalization", "config": {"name": "normalization", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}}, {"class_name": "PruneLowMagnitude", "config": {"name": "prune_low_magnitude_conv2d", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [8, 10]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu6", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "pruning_schedule": {"class_name": "ConstantSparsity", "config": {"target_sparsity": 0.5, "begin_step": 0, "end_step": -1, "frequency": 100}}, "block_size": {"class_name": "__tuple__", "items": [1, 1]}, "block_pooling_type": "AVG"}}, {"class_name": "PruneLowMagnitude", "config": {"name": "prune_low_magnitude_max_pooling2d", "trainable": true, "dtype": "float32", "layer": {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}}, "pruning_schedule": {"class_name": "ConstantSparsity", "config": {"target_sparsity": 0.5, "begin_step": 0, "end_step": -1, "frequency": 100}}, "block_size": {"class_name": "__tuple__", "items": [1, 1]}, "block_pooling_type": "AVG"}}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}}, {"class_name": "PruneLowMagnitude", "config": {"name": "prune_low_magnitude_flatten", "trainable": true, "dtype": "float32", "layer": {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}}, "pruning_schedule": {"class_name": "ConstantSparsity", "config": {"target_sparsity": 0.5, "begin_step": 0, "end_step": -1, "frequency": 100}}, "block_size": {"class_name": "__tuple__", "items": [1, 1]}, "block_pooling_type": "AVG"}}, {"class_name": "PruneLowMagnitude", "config": {"name": "prune_low_magnitude_dense", "trainable": true, "dtype": "float32", "layer": {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, "pruning_schedule": {"class_name": "ConstantSparsity", "config": {"target_sparsity": 0.5, "begin_step": 0, "end_step": -1, "frequency": 100}}, "block_size": {"class_name": "__tuple__", "items": [1, 1]}, "block_pooling_type": "AVG"}}]}, "shared_object_id": 15, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40, 49, 1]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 40, 49, 1]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "ExtAudioDataModelPrun", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 40, 49, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "shared_object_id": 0}, {"class_name": "Normalization", "config": {"name": "normalization", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "shared_object_id": 1}, {"class_name": "PruneLowMagnitude", "config": {"name": "prune_low_magnitude_conv2d", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [8, 10]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu6", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4}, "pruning_schedule": {"class_name": "ConstantSparsity", "config": {"target_sparsity": 0.5, "begin_step": 0, "end_step": -1, "frequency": 100}}, "block_size": {"class_name": "__tuple__", "items": [1, 1]}, "block_pooling_type": "AVG"}, "shared_object_id": 5}, {"class_name": "PruneLowMagnitude", "config": {"name": "prune_low_magnitude_max_pooling2d", "trainable": true, "dtype": "float32", "layer": {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 6}, "pruning_schedule": {"class_name": "ConstantSparsity", "config": {"target_sparsity": 0.5, "begin_step": 0, "end_step": -1, "frequency": 100}}, "block_size": {"class_name": "__tuple__", "items": [1, 1]}, "block_pooling_type": "AVG"}, "shared_object_id": 7}, {"class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "shared_object_id": 8}, {"class_name": "PruneLowMagnitude", "config": {"name": "prune_low_magnitude_flatten", "trainable": true, "dtype": "float32", "layer": {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 9}, "pruning_schedule": {"class_name": "ConstantSparsity", "config": {"target_sparsity": 0.5, "begin_step": 0, "end_step": -1, "frequency": 100}}, "block_size": {"class_name": "__tuple__", "items": [1, 1]}, "block_pooling_type": "AVG"}, "shared_object_id": 10}, {"class_name": "PruneLowMagnitude", "config": {"name": "prune_low_magnitude_dense", "trainable": true, "dtype": "float32", "layer": {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 13}, "pruning_schedule": {"class_name": "ConstantSparsity", "config": {"target_sparsity": 0.5, "begin_step": 0, "end_step": -1, "frequency": 100}}, "block_size": {"class_name": "__tuple__", "items": [1, 1]}, "block_pooling_type": "AVG"}, "shared_object_id": 14}]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": true}, "shared_object_id": 16}, "metrics": [[{"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 17}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0009964072378352284, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?

_keep_axis
_reduce_axis
_reduce_axis_mask
_broadcast_shape
mean
variance
	count
	keras_api
?_adapt_function"?
_tf_keras_layer?{"name": "normalization", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, null]}, "stateful": true, "must_restore_from_config": true, "class_name": "Normalization", "config": {"name": "normalization", "trainable": true, "batch_input_shape": {"class_name": "__tuple__", "items": [null, null, null, null]}, "dtype": "float32", "axis": {"class_name": "__tuple__", "items": [-1]}}, "shared_object_id": 1, "build_input_shape": [null, 40, 49, 1]}
?
pruning_vars
	layer
prunable_weights
mask
	threshold
pruning_step
trainable_variables
regularization_losses
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?
{"name": "prune_low_magnitude_conv2d", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "PruneLowMagnitude", "config": {"name": "prune_low_magnitude_conv2d", "trainable": true, "dtype": "float32", "layer": {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [8, 10]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu6", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4}, "pruning_schedule": {"class_name": "ConstantSparsity", "config": {"target_sparsity": 0.5, "begin_step": 0, "end_step": -1, "frequency": 100}}, "block_size": {"class_name": "__tuple__", "items": [1, 1]}, "block_pooling_type": "AVG"}, "shared_object_id": 5, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40, 49, 1]}}
?	
pruning_vars
	 layer
!prunable_weights
"pruning_step
#trainable_variables
$regularization_losses
%	variables
&	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "prune_low_magnitude_max_pooling2d", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "PruneLowMagnitude", "config": {"name": "prune_low_magnitude_max_pooling2d", "trainable": true, "dtype": "float32", "layer": {"class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 6}, "pruning_schedule": {"class_name": "ConstantSparsity", "config": {"target_sparsity": 0.5, "begin_step": 0, "end_step": -1, "frequency": 100}}, "block_size": {"class_name": "__tuple__", "items": [1, 1]}, "block_pooling_type": "AVG"}, "shared_object_id": 7, "build_input_shape": {"class_name": "TensorShape", "items": [null, 17, 20, 8]}}
?
'trainable_variables
(regularization_losses
)	variables
*	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dropout", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dropout", "config": {"name": "dropout", "trainable": true, "dtype": "float32", "rate": 0.25, "noise_shape": null, "seed": null}, "shared_object_id": 8}
?
+pruning_vars
	,layer
-prunable_weights
.pruning_step
/trainable_variables
0regularization_losses
1	variables
2	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "prune_low_magnitude_flatten", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "PruneLowMagnitude", "config": {"name": "prune_low_magnitude_flatten", "trainable": true, "dtype": "float32", "layer": {"class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 9}, "pruning_schedule": {"class_name": "ConstantSparsity", "config": {"target_sparsity": 0.5, "begin_step": 0, "end_step": -1, "frequency": 100}}, "block_size": {"class_name": "__tuple__", "items": [1, 1]}, "block_pooling_type": "AVG"}, "shared_object_id": 10, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 10, 8]}}
?
3pruning_vars
	4layer
5prunable_weights
6mask
7	threshold
8pruning_step
9trainable_variables
:regularization_losses
;	variables
<	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"name": "prune_low_magnitude_dense", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "PruneLowMagnitude", "config": {"name": "prune_low_magnitude_dense", "trainable": true, "dtype": "float32", "layer": {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 13}, "pruning_schedule": {"class_name": "ConstantSparsity", "config": {"target_sparsity": 0.5, "begin_step": 0, "end_step": -1, "frequency": 100}}, "block_size": {"class_name": "__tuple__", "items": [1, 1]}, "block_pooling_type": "AVG"}, "shared_object_id": 14, "build_input_shape": {"class_name": "TensorShape", "items": [null, 640]}}
?
=iter

>beta_1

?beta_2
	@decay
Alearning_rateBm?Cm?Dm?Em?Bv?Cv?Dv?Ev?"
	optimizer
<
B0
C1
D2
E3"
trackable_list_wrapper
 "
trackable_list_wrapper
?
0
1
2
B3
C4
5
6
7
"8
.9
D10
E11
612
713
814"
trackable_list_wrapper
?
Fmetrics

Glayers
trainable_variables
Hnon_trainable_variables
Ilayer_regularization_losses
	regularization_losses
Jlayer_metrics

	variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
:2mean
:2variance
:	 2count
"
_generic_user_object
'
K0"
trackable_list_wrapper
?


Bkernel
Cbias
Ltrainable_variables
Mregularization_losses
N	variables
O	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?	{"name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 8, "kernel_size": {"class_name": "__tuple__", "items": [8, 10]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu6", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 2}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 3}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}, "shared_object_id": 18}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 40, 49, 1]}}
'
B0"
trackable_list_wrapper
9:7
(2prune_low_magnitude_conv2d/mask
.:, (2$prune_low_magnitude_conv2d/threshold
/:-	 2'prune_low_magnitude_conv2d/pruning_step
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
C
B0
C1
2
3
4"
trackable_list_wrapper
?
Pmetrics

Qlayers
trainable_variables
Rlayer_regularization_losses
Snon_trainable_variables
regularization_losses
Tlayer_metrics
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
?
Utrainable_variables
Vregularization_losses
W	variables
X	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "max_pooling2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling2d", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "shared_object_id": 6, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 19}}
 "
trackable_list_wrapper
6:4	 2.prune_low_magnitude_max_pooling2d/pruning_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
"0"
trackable_list_wrapper
?
Ymetrics

Zlayers
#trainable_variables
[layer_regularization_losses
\non_trainable_variables
$regularization_losses
]layer_metrics
%	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
^metrics

_layers
'trainable_variables
`layer_regularization_losses
anon_trainable_variables
(regularization_losses
blayer_metrics
)	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
?
ctrainable_variables
dregularization_losses
e	variables
f	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "shared_object_id": 9, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 20}}
 "
trackable_list_wrapper
0:.	 2(prune_low_magnitude_flatten/pruning_step
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
.0"
trackable_list_wrapper
?
gmetrics

hlayers
/trainable_variables
ilayer_regularization_losses
jnon_trainable_variables
0regularization_losses
klayer_metrics
1	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
'
l0"
trackable_list_wrapper
?

Dkernel
Ebias
mtrainable_variables
nregularization_losses
o	variables
p	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 4, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 11}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 12}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "shared_object_id": 13, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 640}}, "shared_object_id": 21}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 640]}}
'
D0"
trackable_list_wrapper
1:/	?(2prune_low_magnitude_dense/mask
-:+ (2#prune_low_magnitude_dense/threshold
.:,	 2&prune_low_magnitude_dense/pruning_step
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
C
D0
E1
62
73
84"
trackable_list_wrapper
?
qmetrics

rlayers
9trainable_variables
slayer_regularization_losses
tnon_trainable_variables
:regularization_losses
ulayer_metrics
;	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2iter
: (2beta_1
: (2beta_2
: (2decay
: (2learning_rate
':%
2conv2d/kernel
:2conv2d/bias
:	?2dense/kernel
:2
dense/bias
.
v0
w1"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
n
0
1
2
3
4
5
"6
.7
68
79
810"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
6
B0
1
2"
trackable_tuple_wrapper
.
B0
C1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
B0
C1"
trackable_list_wrapper
?
xmetrics

ylayers
Ltrainable_variables
zlayer_regularization_losses
{non_trainable_variables
Mregularization_losses
|layer_metrics
N	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
}metrics

~layers
Utrainable_variables
layer_regularization_losses
?non_trainable_variables
Vregularization_losses
?layer_metrics
W	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
 0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
"0"
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
?
?metrics
?layers
ctrainable_variables
 ?layer_regularization_losses
?non_trainable_variables
dregularization_losses
?layer_metrics
e	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
,0"
trackable_list_wrapper
 "
trackable_list_wrapper
'
.0"
trackable_list_wrapper
 "
trackable_dict_wrapper
6
D0
61
72"
trackable_tuple_wrapper
.
D0
E1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
D0
E1"
trackable_list_wrapper
?
?metrics
?layers
mtrainable_variables
 ?layer_regularization_losses
?non_trainable_variables
nregularization_losses
?layer_metrics
o	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
'
40"
trackable_list_wrapper
 "
trackable_list_wrapper
5
60
71
82"
trackable_list_wrapper
 "
trackable_dict_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 22}
?

?total

?count
?
_fn_kwargs
?	variables
?	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 17}
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
 "
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
,:*
2Adam/conv2d/kernel/m
:2Adam/conv2d/bias/m
$:"	?2Adam/dense/kernel/m
:2Adam/dense/bias/m
,:*
2Adam/conv2d/kernel/v
:2Adam/conv2d/bias/v
$:"	?2Adam/dense/kernel/v
:2Adam/dense/bias/v
?2?
6__inference_ExtAudioDataModelPrun_layer_call_fn_246212
6__inference_ExtAudioDataModelPrun_layer_call_fn_246976
6__inference_ExtAudioDataModelPrun_layer_call_fn_247009
6__inference_ExtAudioDataModelPrun_layer_call_fn_246846?
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
!__inference__wrapped_model_246093?
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
?2?
Q__inference_ExtAudioDataModelPrun_layer_call_and_return_conditional_losses_247051
Q__inference_ExtAudioDataModelPrun_layer_call_and_return_conditional_losses_247458
Q__inference_ExtAudioDataModelPrun_layer_call_and_return_conditional_losses_246880
Q__inference_ExtAudioDataModelPrun_layer_call_and_return_conditional_losses_246926?
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
__inference_adapt_step_247504?
???
FullArgSpec
args?

jiterator
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
;__inference_prune_low_magnitude_conv2d_layer_call_fn_247515
;__inference_prune_low_magnitude_conv2d_layer_call_fn_247530?
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
V__inference_prune_low_magnitude_conv2d_layer_call_and_return_conditional_losses_247545
V__inference_prune_low_magnitude_conv2d_layer_call_and_return_conditional_losses_247674?
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
B__inference_prune_low_magnitude_max_pooling2d_layer_call_fn_247679
B__inference_prune_low_magnitude_max_pooling2d_layer_call_fn_247686?
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
]__inference_prune_low_magnitude_max_pooling2d_layer_call_and_return_conditional_losses_247691
]__inference_prune_low_magnitude_max_pooling2d_layer_call_and_return_conditional_losses_247761?
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
(__inference_dropout_layer_call_fn_247766
(__inference_dropout_layer_call_fn_247771?
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
C__inference_dropout_layer_call_and_return_conditional_losses_247776
C__inference_dropout_layer_call_and_return_conditional_losses_247788?
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
<__inference_prune_low_magnitude_flatten_layer_call_fn_247793
<__inference_prune_low_magnitude_flatten_layer_call_fn_247800?
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
W__inference_prune_low_magnitude_flatten_layer_call_and_return_conditional_losses_247806
W__inference_prune_low_magnitude_flatten_layer_call_and_return_conditional_losses_247877?
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
:__inference_prune_low_magnitude_dense_layer_call_fn_247888
:__inference_prune_low_magnitude_dense_layer_call_fn_247903?
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
U__inference_prune_low_magnitude_dense_layer_call_and_return_conditional_losses_247917
U__inference_prune_low_magnitude_dense_layer_call_and_return_conditional_losses_248045?
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
$__inference_signature_wrapper_246955input_1"?
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
.__inference_max_pooling2d_layer_call_fn_246105?
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
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_246099?
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
Q__inference_ExtAudioDataModelPrun_layer_call_and_return_conditional_losses_246880sBCD6E@?=
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
Q__inference_ExtAudioDataModelPrun_layer_call_and_return_conditional_losses_246926yBC".8D67E@?=
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
Q__inference_ExtAudioDataModelPrun_layer_call_and_return_conditional_losses_247051rBCD6E??<
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
Q__inference_ExtAudioDataModelPrun_layer_call_and_return_conditional_losses_247458xBC".8D67E??<
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
6__inference_ExtAudioDataModelPrun_layer_call_fn_246212fBCD6E@?=
6?3
)?&
input_1?????????(1
p 

 
? "???????????
6__inference_ExtAudioDataModelPrun_layer_call_fn_246846lBC".8D67E@?=
6?3
)?&
input_1?????????(1
p

 
? "???????????
6__inference_ExtAudioDataModelPrun_layer_call_fn_246976eBCD6E??<
5?2
(?%
inputs?????????(1
p 

 
? "???????????
6__inference_ExtAudioDataModelPrun_layer_call_fn_247009kBC".8D67E??<
5?2
(?%
inputs?????????(1
p

 
? "???????????
!__inference__wrapped_model_246093?BCD6E8?5
.?+
)?&
input_1?????????(1
? "U?R
P
prune_low_magnitude_dense3?0
prune_low_magnitude_dense?????????u
__inference_adapt_step_247504TI?F
??<
:?7%?"
 ??????????(1IteratorSpec
? "
 ?
C__inference_dropout_layer_call_and_return_conditional_losses_247776l;?8
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
C__inference_dropout_layer_call_and_return_conditional_losses_247788l;?8
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
(__inference_dropout_layer_call_fn_247766_;?8
1?.
(?%
inputs?????????

p 
? " ??????????
?
(__inference_dropout_layer_call_fn_247771_;?8
1?.
(?%
inputs?????????

p
? " ??????????
?
I__inference_max_pooling2d_layer_call_and_return_conditional_losses_246099?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
.__inference_max_pooling2d_layer_call_fn_246105?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
V__inference_prune_low_magnitude_conv2d_layer_call_and_return_conditional_losses_247545qBC;?8
1?.
(?%
inputs?????????(1
p 
? "-?*
#? 
0?????????
? ?
V__inference_prune_low_magnitude_conv2d_layer_call_and_return_conditional_losses_247674sBC;?8
1?.
(?%
inputs?????????(1
p
? "-?*
#? 
0?????????
? ?
;__inference_prune_low_magnitude_conv2d_layer_call_fn_247515dBC;?8
1?.
(?%
inputs?????????(1
p 
? " ???????????
;__inference_prune_low_magnitude_conv2d_layer_call_fn_247530fBC;?8
1?.
(?%
inputs?????????(1
p
? " ???????????
U__inference_prune_low_magnitude_dense_layer_call_and_return_conditional_losses_247917bD6E4?1
*?'
!?
inputs??????????
p 
? "%?"
?
0?????????
? ?
U__inference_prune_low_magnitude_dense_layer_call_and_return_conditional_losses_248045d8D67E4?1
*?'
!?
inputs??????????
p
? "%?"
?
0?????????
? ?
:__inference_prune_low_magnitude_dense_layer_call_fn_247888UD6E4?1
*?'
!?
inputs??????????
p 
? "???????????
:__inference_prune_low_magnitude_dense_layer_call_fn_247903W8D67E4?1
*?'
!?
inputs??????????
p
? "???????????
W__inference_prune_low_magnitude_flatten_layer_call_and_return_conditional_losses_247806e;?8
1?.
(?%
inputs?????????

p 
? "&?#
?
0??????????
? ?
W__inference_prune_low_magnitude_flatten_layer_call_and_return_conditional_losses_247877h.;?8
1?.
(?%
inputs?????????

p
? "&?#
?
0??????????
? ?
<__inference_prune_low_magnitude_flatten_layer_call_fn_247793X;?8
1?.
(?%
inputs?????????

p 
? "????????????
<__inference_prune_low_magnitude_flatten_layer_call_fn_247800[.;?8
1?.
(?%
inputs?????????

p
? "????????????
]__inference_prune_low_magnitude_max_pooling2d_layer_call_and_return_conditional_losses_247691l;?8
1?.
(?%
inputs?????????
p 
? "-?*
#? 
0?????????

? ?
]__inference_prune_low_magnitude_max_pooling2d_layer_call_and_return_conditional_losses_247761o";?8
1?.
(?%
inputs?????????
p
? "-?*
#? 
0?????????

? ?
B__inference_prune_low_magnitude_max_pooling2d_layer_call_fn_247679_;?8
1?.
(?%
inputs?????????
p 
? " ??????????
?
B__inference_prune_low_magnitude_max_pooling2d_layer_call_fn_247686b";?8
1?.
(?%
inputs?????????
p
? " ??????????
?
$__inference_signature_wrapper_246955?BCD6EC?@
? 
9?6
4
input_1)?&
input_1?????????(1"U?R
P
prune_low_magnitude_dense3?0
prune_low_magnitude_dense?????????