"?P
DDeviceIDLE"IDLE1?????ƠBA?????ƠBQ      ??Y      ???Unknown
BHostIDLE"IDLE1?O????@A?O????@a????)~??i????)~???Unknown
?HostConv2DBackpropFilter"Bgradient_tape/ExtAudioDataModel/conv2d/Conv2D/Conv2DBackpropFilter(1H?z.!?@9H?z.!?@AH?z.!?@IH?z.!?@aמ?Wm???i>????????Unknown
xHost_FusedConv2D"ExtAudioDataModel/conv2d/Relu6(1???Q??@9???Q??@A???Q??@I???Q??@an?w????i??-$????Unknown
?Host	Relu6Grad"6gradient_tape/ExtAudioDataModel/conv2d/Relu6/Relu6Grad(1?"??~J~@9?"??~J~@A?"??~J~@I?"??~J~@a??9m%??i?Ү?Ob???Unknown
|HostMaxPool"'ExtAudioDataModel/max_pooling2d/MaxPool(1???K7z@9???K7z@A???K7z@I???K7z@a??t{H>??i?x??A$???Unknown
?HostBiasAddGrad":gradient_tape/ExtAudioDataModel/conv2d/BiasAdd/BiasAddGrad(1Zd;??x@9Zd;??x@AZd;??x@IZd;??x@a?e6m1??ih?=D?????Unknown
?HostMaxPoolGrad"Agradient_tape/ExtAudioDataModel/max_pooling2d/MaxPool/MaxPoolGrad(1??/??p@9??/??p@A??/??p@I??/??p@aw??#?8??ib???Z???Unknown
?	HostRandomUniform">ExtAudioDataModel/dropout/dropout/random_uniform/RandomUniform(1?A`?кb@9?A`?кb@A?A`?кb@I?A`?кb@a?~?Og??i\p/M????Unknown
?
HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1?l????a@9?l????a@A?l????a@I?l????a@a9?t??V??iu??٨????Unknown
?HostMatMul".gradient_tape/ExtAudioDataModel/dense/MatMul_1(1     Da@9     Da@A     Da@I     Da@au?????iG{-??!???Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1X9??T@9X9??T@AX9??T@IX9??T@a???ƛr?i?G?G???Unknown
tHostSub"#ExtAudioDataModel/normalization/sub(1?????kQ@9?????kQ@A?????kQ@I?????kQ@a.k???/p?ic?R?lg???Unknown
|HostRealDiv"'ExtAudioDataModel/normalization/truediv(1\???(?P@9\???(?P@A\???(?P@I\???(?P@a3?B?o?izŕ&?????Unknown
xHostCast"&ExtAudioDataModel/dropout/dropout/Cast(1??Q??N@9??Q??N@A??Q??N@I??Q??N@aSG?k?l?i?K?????Unknown
?HostMatMul",gradient_tape/ExtAudioDataModel/dense/MatMul(1?G?zdH@9?G?zdH@A?G?zdH@I?G?zdH@a?SOT?f?i????????Unknown
?HostReadVariableOp".ExtAudioDataModel/conv2d/Conv2D/ReadVariableOp(1!?rh?-E@9!?rh?-E@A!?rh?-E@I!?rh?-E@a.?Z??c?i???o????Unknown
yHost_FusedMatMul"ExtAudioDataModel/dense/BiasAdd(1?A`?ТD@9?A`?ТD@A?A`?ТD@I?A`?ТD@a?????,c?i*??W?????Unknown
?HostGreaterEqual".ExtAudioDataModel/dropout/dropout/GreaterEqual(1!?rh??C@9!?rh??C@A!?rh??C@I!?rh??C@a?K??#+b?ivy?{?????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1-??臨B@9-??臨B@A-??臨B@I-??臨B@a?zBrca?i????*???Unknown?
oHostReadVariableOp"Adam/ReadVariableOp(1?G?znB@9?G?znB@A?G?znB@I?G?znB@a1?3 a?i??)?J???Unknown
?HostDataset"0Iterator::Model::MaxIntraOpParallelism::Prefetch(1V-??@@9V-??@@AV-??@@IV-??@@aǒ??^?i?,?q?$???Unknown
{HostDataset"&Iterator::Model::MaxIntraOpParallelism(1??n?0N@9??n?0N@A??ʡ%;@I??ʡ%;@a?#MH?9Y?i8S[8X1???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1^?Ik:@9^?Ik:@A^?Ik:@I^?Ik:@a4k?`$?X?i?ŋJ?=???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1w??/?9@9w??/?9@Aw??/?9@Iw??/?9@aUݪM??W?i]????I???Unknown
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1?S㥛d9@9?S㥛d9@A?S㥛d9@I?S㥛d9@a??N_R?W?i?B??PU???Unknown
iHostWriteSummary"WriteSummary(1?t??2@9?t??2@A?t??2@I?t??2@a\\?EQ?i?p?t?]???Unknown?
xHostMul"'ExtAudioDataModel/dropout/dropout/Mul_1(1??????0@9??????0@A??????0@I??????0@ai?qf?N?i??sΩe???Unknown
?HostMul"5gradient_tape/ExtAudioDataModel/dropout/dropout/Mul_1(1??K7?0@9??K7?0@A??K7?0@I??K7?0@a?tξM?i*??m???Unknown
gHostStridedSlice"strided_slice(1Zd;?O?.@9Zd;?O?.@AZd;?O?.@IZd;?O?.@a?????L?i?t?@At???Unknown
?HostDataset"=Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCache(1sh??|?2@9sh??|?2@A+??.@I+??.@aԟ?k ?K?i?V??<{???Unknown
Z HostArgMax"ArgMax(1?I+?*@9?I+?*@A?I+?*@I?I+?*@a(?o?N?H?i???Tf????Unknown
V!HostSum"Sum_2(1??~j??)@9??~j??)@A??~j??)@I??~j??)@a4y??>H?i?Z<?m????Unknown
?"HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1j?t?'@9j?t?'@Aj?t?'@Ij?t?'@a?V~?bE?i???ƌ???Unknown
?#HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1?&1??%@9?&1??%@A?&1??%@I?&1??%@a????cPD?i;옶ڑ???Unknown
[$HostAddV2"Adam/add(1??"??~$@9??"??~$@A??"??~$@I??"??~$@aE???tC?i斻??????Unknown
?%HostBiasAddGrad"9gradient_tape/ExtAudioDataModel/dense/BiasAdd/BiasAddGrad(1?????K$@9?????K$@A?????K$@I?????K$@a?#k??B?i?_?T????Unknown
b&HostDivNoNan"div_no_nan_1(1     ?#@9     ?#@A     ?#@I     ?#@a0,!??B?i??6.ܟ???Unknown
l'HostIteratorGetNext"IteratorGetNext(133333?!@933333?!@A33333?!@I33333?!@a???Mԭ@?i?YJ?????Unknown
v(HostSqrt"$ExtAudioDataModel/normalization/Sqrt(1?A`???!@9?A`???!@A?A`???!@I?A`???!@a?????@?iZ	J/????Unknown
t)HostAssignAddVariableOp"AssignAddVariableOp(1%??C?!@9%??C?!@A%??C?!@I%??C?!@a?pv???@?i??.xQ????Unknown
v*HostAssignAddVariableOp"AssignAddVariableOp_2(1
ףp=? @9
ףp=? @A
ףp=? @I
ףp=? @aG???ڼ>?iW
?)????Unknown
`+HostDivNoNan"
div_no_nan(1q=
ףp @9q=
ףp @Aq=
ףp @Iq=
ףp @a?/?G?>?i,???????Unknown
v,HostMul"%ExtAudioDataModel/dropout/dropout/Mul(1;?O???@9;?O???@A;?O???@I;?O???@ap??'?:?i?ȇW????Unknown
w-HostReadVariableOp"div_no_nan_1/ReadVariableOp(1?E????@9?E????@A?E????@I?E????@a?<#e^?:?ifmTM?????Unknown
?.HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1Zd;?O@9Zd;?O@AZd;?O@IZd;?O@aЌ???:?i??g?????Unknown
Y/HostPow"Adam/Pow(1?????M@9?????M@A?????M@I?????M@aUZ???^9?iC?D????Unknown
?0HostReadVariableOp"6ExtAudioDataModel/normalization/Reshape/ReadVariableOp(1?????@9?????@A?????@I?????@a|Q????8?i????;????Unknown
?1HostReadVariableOp"8ExtAudioDataModel/normalization/Reshape_1/ReadVariableOp(1P??n?@9P??n?@AP??n?@IP??n?@a??X???7?i??2????Unknown
?2HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1ˡE??}@9ˡE??}@AˡE??}@IˡE??}@a^_?	߯7?i7X??(????Unknown
d3HostDataset"Iterator::Model(1??v???P@9??v???P@A???S?%@I???S?%@a??H?
^7?iJ?[K????Unknown
?4HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1?I+?@9?I+?@A?I+?@I?I+?@a??~??5?i1]??????Unknown
[5HostPow"
Adam/Pow_1(1?|?5^:@9?|?5^:@A?|?5^:@I?|?5^:@a?93]S?5?i??ȋ?????Unknown
?6HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1y?&1,@9y?&1,@Ay?&1,@Iy?&1,@am?!?4?i????????Unknown
?7HostReadVariableOp".ExtAudioDataModel/dense/BiasAdd/ReadVariableOp(1???Mb?@9???Mb?@A???Mb?@I???Mb?@a???q?3?i$	*Dy????Unknown
e8Host
LogicalAnd"
LogicalAnd(1`??"?y@9`??"?y@A`??"?y@I`??"?y@a???3?i?,~?????Unknown?
?9HostReadVariableOp"-ExtAudioDataModel/dense/MatMul/ReadVariableOp(1X9??v>@9X9??v>@AX9??v>@IX9??v>@a??؂?2?i)C?
4????Unknown
u:HostReadVariableOp"div_no_nan/ReadVariableOp(1??Q?@9??Q?@A??Q?@I??Q?@aq??,$?1?i?^?l????Unknown
X;HostEqual"Equal(1T㥛? @9T㥛? @AT㥛? @IT㥛? @am2?+?0?ij?Ք?????Unknown
v<HostAssignAddVariableOp"AssignAddVariableOp_4(1???(\@9???(\@A???(\@I???(\@a??Vi>?/?iՒ?؂????Unknown
~=HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1j?t?@9j?t?@Aj?t?@Ij?t?@a????O?/?iR!??|????Unknown
|>HostMaximum"'ExtAudioDataModel/normalization/Maximum(1???Sc@9???Sc@A???Sc@I???Sc@aI.??t.?i4dad????Unknown
??HostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1X9??v>@9X9??v>@AX9??v>@IX9??v>@a??7?0.?i???G????Unknown
[@HostCast"	Adam/Cast(1j?t?@9j?t?@Aj?t?@Ij?t?@aH[???+?i??W????Unknown
VAHostCast"Cast(1????K7@9????K7@A????K7@I????K7@a?O?%+?i????????Unknown
?BHostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1!?rh??@9!?rh??@A!?rh??@I!?rh??@a?$?T?)?i???V????Unknown
XCHostCast"Cast_2(1???x?&@9???x?&@A???x?&@I???x?&@aֆ?½:)?iϧ??????Unknown
?DHostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1?|?5^?
@9?|?5^?
@A?|?5^?
@I?|?5^?
@a?????(?i???w????Unknown
vEHostAssignAddVariableOp"AssignAddVariableOp_3(1????Mb
@9????Mb
@A????Mb
@I????Mb
@a?t??(?i???2 ????Unknown
XFHostCast"Cast_4(1ףp=
?	@9ףp=
?	@Aףp=
?	@Iףp=
?	@a?rj??(?iy\]?????Unknown
?GHostDataset"AIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl(1??n??	@9??n??	@A??n??	@I??n??	@a??????'?i3??8?????Unknown
vHHostAssignAddVariableOp"AssignAddVariableOp_1(1?|?5^?@9?|?5^?@A?|?5^?@I?|?5^?@a4???"?&?i?V??n????Unknown
XIHostCast"Cast_3(1ffffff@9ffffff@Affffff@Iffffff@a??N?]?$?i????????Unknown
wJHostReadVariableOp"div_no_nan/ReadVariableOp_1(1^?I+@9^?I+@A^?I+@I^?I+@a ?>eN?#?i?_???????Unknown
zKHostReadVariableOp"Adam/Identity_1/ReadVariableOp(1V-???@9V-???@AV-???@IV-???@a?J?a_?!?icxv????Unknown
?LHostReadVariableOp"/ExtAudioDataModel/conv2d/BiasAdd/ReadVariableOp(1?Q???@9?Q???@A?Q???@I?Q???@a$>ro?? ?i?o??????Unknown
TMHostMul"Mul(1h??|?5@9h??|?5@Ah??|?5@Ih??|?5@ae??????i???[????Unknown
zNHostReadVariableOp"Adam/Identity_2/ReadVariableOp(1??v??@9??v??@A??v??@I??v??@a???9-??i?Y3?????Unknown
?OHostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1?t?V??9?t?V??A?t?V??I?t?V??a?@?f!0?i??>&?????Unknown
yPHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1?G?z???9?G?z???A?G?z???I?G?z???a?6????iM????????Unknown
xQHostReadVariableOp"Adam/Identity/ReadVariableOp(1?????K??9?????K??A?????K??I?????K??a?????\?i?i㮡????Unknown
aRHostIdentity"Identity(1?K7?A`??9?K7?A`??A?K7?A`??I?K7?A`??a?X?%G??i     ???Unknown?*?O
?HostConv2DBackpropFilter"Bgradient_tape/ExtAudioDataModel/conv2d/Conv2D/Conv2DBackpropFilter(1H?z.!?@9H?z.!?@AH?z.!?@IH?z.!?@a 5?z???i 5?z????Unknown
xHost_FusedConv2D"ExtAudioDataModel/conv2d/Relu6(1???Q??@9???Q??@A???Q??@I???Q??@a8?<??p??i<te?????Unknown
?Host	Relu6Grad"6gradient_tape/ExtAudioDataModel/conv2d/Relu6/Relu6Grad(1?"??~J~@9?"??~J~@A?"??~J~@I?"??~J~@a:?FX???i5?e?)???Unknown
|HostMaxPool"'ExtAudioDataModel/max_pooling2d/MaxPool(1???K7z@9???K7z@A???K7z@I???K7z@amǨ?ڰ?ic??z6E???Unknown
?HostBiasAddGrad":gradient_tape/ExtAudioDataModel/conv2d/BiasAdd/BiasAddGrad(1Zd;??x@9Zd;??x@AZd;??x@IZd;??x@a2SY????i?'?P0I???Unknown
?HostMaxPoolGrad"Agradient_tape/ExtAudioDataModel/max_pooling2d/MaxPool/MaxPoolGrad(1??/??p@9??/??p@A??/??p@I??/??p@a?$(????i?b?z????Unknown
?HostRandomUniform">ExtAudioDataModel/dropout/dropout/random_uniform/RandomUniform(1?A`?кb@9?A`?кb@A?A`?кb@I?A`?кb@aƫ???2??iu߷Qf???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1?l????a@9?l????a@A?l????a@I?l????a@a???Z???i??????Unknown
?	HostMatMul".gradient_tape/ExtAudioDataModel/dense/MatMul_1(1     Da@9     Da@A     Da@I     Da@aƓk&?N??i?c??F????Unknown
?
HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1X9??T@9X9??T@AX9??T@IX9??T@a????߉?iX?k?5???Unknown
tHostSub"#ExtAudioDataModel/normalization/sub(1?????kQ@9?????kQ@A?????kQ@I?????kQ@a?	??????i<\?C͏???Unknown
|HostRealDiv"'ExtAudioDataModel/normalization/truediv(1\???(?P@9\???(?P@A\???(?P@I\???(?P@a??W???i<?9?I????Unknown
xHostCast"&ExtAudioDataModel/dropout/dropout/Cast(1??Q??N@9??Q??N@A??Q??N@I??Q??N@a8?pl܃?i???:?5???Unknown
?HostMatMul",gradient_tape/ExtAudioDataModel/dense/MatMul(1?G?zdH@9?G?zdH@A?G?zdH@I?G?zdH@a8P?AŃ?iY????t???Unknown
?HostReadVariableOp".ExtAudioDataModel/conv2d/Conv2D/ReadVariableOp(1!?rh?-E@9!?rh?-E@A!?rh?-E@I!?rh?-E@a???\{?i'??{????Unknown
yHost_FusedMatMul"ExtAudioDataModel/dense/BiasAdd(1?A`?ТD@9?A`?ТD@A?A`?ТD@I?A`?ТD@a???P?z?i?????????Unknown
?HostGreaterEqual".ExtAudioDataModel/dropout/dropout/GreaterEqual(1!?rh??C@9!?rh??C@A!?rh??C@I!?rh??C@a??Cy?i????T???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1-??臨B@9-??臨B@A-??臨B@I-??臨B@a%lC?s-x?i?7???C???Unknown?
oHostReadVariableOp"Adam/ReadVariableOp(1?G?znB@9?G?znB@A?G?znB@I?G?znB@aYd???w?iI ?Os???Unknown
?HostDataset"0Iterator::Model::MaxIntraOpParallelism::Prefetch(1V-??@@9V-??@@AV-??@@IV-??@@aH`o?wu?i??~}>????Unknown
{HostDataset"&Iterator::Model::MaxIntraOpParallelism(1??n?0N@9??n?0N@A??ʡ%;@I??ʡ%;@a??oh?q?iF@]NQ????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1^?Ik:@9^?Ik:@A^?Ik:@I^?Ik:@aE7j??q?i??s????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1w??/?9@9w??/?9@Aw??/?9@Iw??/?9@aKM???p?iЮ2Ɋ???Unknown
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1?S㥛d9@9?S㥛d9@A?S㥛d9@I?S㥛d9@a?C?Wgp?iV6GxY%???Unknown
iHostWriteSummary"WriteSummary(1?t??2@9?t??2@A?t??2@I?t??2@a??CHh?iE ??\=???Unknown?
xHostMul"'ExtAudioDataModel/dropout/dropout/Mul_1(1??????0@9??????0@A??????0@I??????0@a???jre?i?<+?R???Unknown
?HostMul"5gradient_tape/ExtAudioDataModel/dropout/dropout/Mul_1(1??K7?0@9??K7?0@A??K7?0@I??K7?0@aH*????d?i??}g???Unknown
gHostStridedSlice"strided_slice(1Zd;?O?.@9Zd;?O?.@AZd;?O?.@IZd;?O?.@a*8????c?i*???b{???Unknown
?HostDataset"=Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCache(1sh??|?2@9sh??|?2@A+??.@I+??.@a????jc?i8?l?͎???Unknown
ZHostArgMax"ArgMax(1?I+?*@9?I+?*@A?I+?*@I?I+?*@a6??
#a?in???????Unknown
VHostSum"Sum_2(1??~j??)@9??~j??)@A??~j??)@I??~j??)@a??U?r?`?i?E?????Unknown
? HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1j?t?'@9j?t?'@Aj?t?'@Ij?t?'@ah?_?~?]?i?:??????Unknown
?!HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1?&1??%@9?&1??%@A?&1??%@I?&1??%@a}=??>\?iu???????Unknown
["HostAddV2"Adam/add(1??"??~$@9??"??~$@A??"??~$@I??"??~$@a=?g?{Z?iWށn?????Unknown
?#HostBiasAddGrad"9gradient_tape/ExtAudioDataModel/dense/BiasAdd/BiasAddGrad(1?????K$@9?????K$@A?????K$@I?????K$@a?????8Z?i;???????Unknown
b$HostDivNoNan"div_no_nan_1(1     ?#@9     ?#@A     ?#@I     ?#@a@:?1Y?i[?ç?????Unknown
l%HostIteratorGetNext"IteratorGetNext(133333?!@933333?!@A33333?!@I33333?!@ad3?0W?i?]> ???Unknown
v&HostSqrt"$ExtAudioDataModel/normalization/Sqrt(1?A`???!@9?A`???!@A?A`???!@I?A`???!@a#?]9?W?i??y?????Unknown
t'HostAssignAddVariableOp"AssignAddVariableOp(1%??C?!@9%??C?!@A%??C?!@I%??C?!@a?~rT?V?i???J???Unknown
v(HostAssignAddVariableOp"AssignAddVariableOp_2(1
ףp=? @9
ףp=? @A
ףp=? @I
ףp=? @am??f?^U?i6????!???Unknown
`)HostDivNoNan"
div_no_nan(1q=
ףp @9q=
ףp @Aq=
ףp @Iq=
ףp @a???@=U?iy솲?,???Unknown
v*HostMul"%ExtAudioDataModel/dropout/dropout/Mul(1;?O???@9;?O???@A;?O???@I;?O???@a?M????R?i?Ee??5???Unknown
w+HostReadVariableOp"div_no_nan_1/ReadVariableOp(1?E????@9?E????@A?E????@I?E????@a?N?)?R?i?zA????Unknown
?,HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1Zd;?O@9Zd;?O@AZd;?O@IZd;?O@a?WZHR?i?F??PH???Unknown
Y-HostPow"Adam/Pow(1?????M@9?????M@A?????M@I?????M@a??u%_?Q?i?1@"Q???Unknown
?.HostReadVariableOp"6ExtAudioDataModel/normalization/Reshape/ReadVariableOp(1?????@9?????@A?????@I?????@a?2U?TQ?i??ۓ?Y???Unknown
?/HostReadVariableOp"8ExtAudioDataModel/normalization/Reshape_1/ReadVariableOp(1P??n?@9P??n?@AP??n?@IP??n?@a?t\~{P?iù?
b???Unknown
?0HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1ˡE??}@9ˡE??}@AˡE??}@IˡE??}@auZǶwP?i?f??Ej???Unknown
d1HostDataset"Iterator::Model(1??v???P@9??v???P@A???S?%@I???S?%@a?EkH?>P?i???[er???Unknown
?2HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1?I+?@9?I+?@A?I+?@I?I+?@a۰???eN?iMS??y???Unknown
[3HostPow"
Adam/Pow_1(1?|?5^:@9?|?5^:@A?|?5^:@I?|?5^:@a??VO?N?i?֦x????Unknown
?4HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1y?&1,@9y?&1,@Ay?&1,@Iy?&1,@aa68X?L?iI޴Ψ????Unknown
?5HostReadVariableOp".ExtAudioDataModel/dense/BiasAdd/ReadVariableOp(1???Mb?@9???Mb?@A???Mb?@I???Mb?@a{??w??J?im??.M????Unknown
e6Host
LogicalAnd"
LogicalAnd(1`??"?y@9`??"?y@A`??"?y@I`??"?y@a??c`etJ?i^?*H?????Unknown?
?7HostReadVariableOp"-ExtAudioDataModel/dense/MatMul/ReadVariableOp(1X9??v>@9X9??v>@AX9??v>@IX9??v>@aӕh??'J?i???2t????Unknown
u8HostReadVariableOp"div_no_nan/ReadVariableOp(1??Q?@9??Q?@A??Q?@I??Q?@a8I????H?iwF,?????Unknown
X9HostEqual"Equal(1T㥛? @9T㥛? @AT㥛? @IT㥛? @a?,[??kG?i??t|????Unknown
v:HostAssignAddVariableOp"AssignAddVariableOp_4(1???(\@9???(\@A???(\@I???(\@a~xR??
F?i~?Y??????Unknown
~;HostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1j?t?@9j?t?@Aj?t?@Ij?t?@a??E?i????}????Unknown
|<HostMaximum"'ExtAudioDataModel/normalization/Maximum(1???Sc@9???Sc@A???Sc@I???Sc@a?T?UL,E?iX??ȸ???Unknown
?=HostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1X9??v>@9X9??v>@AX9??v>@IX9??v>@a??H???D?i? ?????Unknown
[>HostCast"	Adam/Cast(1j?t?@9j?t?@Aj?t?@Ij?t?@abt???cC?iq?E??????Unknown
V?HostCast"Cast(1????K7@9????K7@A????K7@I????K7@a?}???B?iP??ݘ????Unknown
?@HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1!?rh??@9!?rh??@A!?rh??@I!?rh??@a?%???
B?i?"??????Unknown
XAHostCast"Cast_2(1???x?&@9???x?&@A???x?&@I???x?&@ab <?A?i?b?~????Unknown
?BHostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1?|?5^?
@9?|?5^?
@A?|?5^?
@I?|?5^?
@a?3	?DA?i??,?????Unknown
vCHostAssignAddVariableOp"AssignAddVariableOp_3(1????Mb
@9????Mb
@A????Mb
@I????Mb
@auC:A?i????????Unknown
XDHostCast"Cast_4(1ףp=
?	@9ףp=
?	@Aףp=
?	@Iףp=
?	@a???C?@?i61?7>????Unknown
?EHostDataset"AIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl(1??n??	@9??n??	@A??n??	@I??n??	@art??@?iS?)?f????Unknown
vFHostAssignAddVariableOp"AssignAddVariableOp_1(1?|?5^?@9?|?5^?@A?|?5^?@I?|?5^?@aS???????i??<e????Unknown
XGHostCast"Cast_3(1ffffff@9ffffff@Affffff@Iffffff@a??}Q??<?ifN?T????Unknown
wHHostReadVariableOp"div_no_nan/ReadVariableOp_1(1^?I+@9^?I+@A^?I+@I^?I+@a??FY;?i?Ŧ}n????Unknown
zIHostReadVariableOp"Adam/Identity_1/ReadVariableOp(1V-???@9V-???@AV-???@IV-???@a
?/Iw8?i??_}????Unknown
?JHostReadVariableOp"/ExtAudioDataModel/conv2d/BiasAdd/ReadVariableOp(1?Q???@9?Q???@A?Q???@I?Q???@a?d?'7?i??1?b????Unknown
TKHostMul"Mul(1h??|?5@9h??|?5@Ah??|?5@Ih??|?5@a?A???;6?i????)????Unknown
zLHostReadVariableOp"Adam/Identity_2/ReadVariableOp(1??v??@9??v??@A??v??@I??v??@a???6?i?4??????Unknown
?MHostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1?t?V??9?t?V??A?t?V??I?t?V??a?pɘ3?i??W?_????Unknown
yNHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1?G?z???9?G?z???A?G?z???I?G?z???ap???L,3?i???u?????Unknown
xOHostReadVariableOp"Adam/Identity/ReadVariableOp(1?????K??9?????K??A?????K??I?????K??a???u?1?iF????????Unknown
aPHostIdentity"Identity(1?K7?A`??9?K7?A`??A?K7?A`??I?K7?A`??a??Շd ?i      ???Unknown?2Nvidia GPU (Maxwell)