"?P
DDeviceIDLE"IDLE1?????ƠBA?????ƠBQ      ??Y      ???Unknown
BHostIDLE"IDLE1ףp=z(?@Aףp=z(?@a?9Q?X%??i?9Q?X%???Unknown
?HostConv2DBackpropFilter"Dgradient_tape/MicroAudioDataModel/conv2d/Conv2D/Conv2DBackpropFilter(1ףp=
ǝ@9ףp=
ǝ@Aףp=
ǝ@Iףp=
ǝ@a?2??WZ??iFକ?????Unknown
zHost_FusedConv2D" MicroAudioDataModel/conv2d/Relu6(1??x馂?@9??x馂?@A??x馂?@I??x馂?@a?Č?p??i???ű????Unknown
?Host	Relu6Grad"8gradient_tape/MicroAudioDataModel/conv2d/Relu6/Relu6Grad(1?l?????@9?l?????@A?l?????@I?l?????@aFf?A????i?R?	j????Unknown
~HostMaxPool")MicroAudioDataModel/max_pooling2d/MaxPool(1-??????@9-??????@A-??????@I-??????@a?v?I?(??i_*y??????Unknown
?HostBiasAddGrad"<gradient_tape/MicroAudioDataModel/conv2d/BiasAdd/BiasAddGrad(1)\???@s@9)\???@s@A)\???@s@I)\???@s@a%?4?&U??i`???g???Unknown
?HostMaxPoolGrad"Cgradient_tape/MicroAudioDataModel/max_pooling2d/MaxPool/MaxPoolGrad(1??n?@s@9??n?@s@A??n?@s@I??n?@s@ay???T??i(s??E????Unknown
?	HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1??S???p@9??S???p@A??S???p@I??S???p@a 㟍j??i@r?7){???Unknown
?
HostRandomUniform"@MicroAudioDataModel/dropout/dropout/random_uniform/RandomUniform(1
ףp=?e@9
ףp=?e@A
ףp=?e@I
ףp=?e@a?`?????i????????Unknown
?HostMatMul"0gradient_tape/MicroAudioDataModel/dense/MatMul_1(1bX9??b@9bX9??b@AbX9??b@IbX9??b@a	?n1?ށ?i$??????Unknown
zHostCast"(MicroAudioDataModel/dropout/dropout/Cast(1^?I?T@9^?I?T@A^?I?T@I^?I?T@a?":?s?ijo? G<???Unknown
~HostRealDiv")MicroAudioDataModel/normalization/truediv(17?A`??S@97?A`??S@A7?A`??S@I7?A`??S@a?F?|?r?i?q?ka???Unknown
?HostMatMul".gradient_tape/MicroAudioDataModel/dense/MatMul(1X9??v^L@9X9??v^L@AX9??v^L@IX9??v^L@a?݆?k?i??{n|???Unknown
vHostSub"%MicroAudioDataModel/normalization/sub(1?A`?ЂK@9?A`?ЂK@A?A`?ЂK@I?A`?ЂK@a?????1j?i??W??????Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1?|?5^JI@9?|?5^JI@A?|?5^JI@I?|?5^JI@aW?U+?h?i[A???????Unknown
?HostGreaterEqual"0MicroAudioDataModel/dropout/dropout/GreaterEqual(1?p=
?F@9?p=
?F@A?p=
?F@I?p=
?F@a.???6?d?iZ?H̪????Unknown
{Host_FusedMatMul"!MicroAudioDataModel/dense/BiasAdd(1ףp=
?E@9ףp=
?E@Aףp=
?E@Iףp=
?E@a?{?|??d?i?B?j9????Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1????K??@9????K??@A????K??@I????K??@as.??o^?iڱPq????Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1?????+?@9?????+?@A?????+?@I?????+?@a???P?]?i*ZRH????Unknown?
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1o??ʁ<@9o??ʁ<@Ao??ʁ<@Io??ʁ<@aVn??$[?i3?P?????Unknown
?HostDataset"0Iterator::Model::MaxIntraOpParallelism::Prefetch(1? ?rh?;@9? ?rh?;@A? ?rh?;@I? ?rh?;@a?[{ݵ|Z?i??????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1?z?G?:@9?z?G?:@A?z?G?:@I?z?G?:@at?)	?Y?i?3D
????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1?/?$?9@9?/?$?9@A?/?$?9@I?/?$?9@a??A???X?irT??9*???Unknown
?HostDataset"=Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCache(1{?G??=@9{?G??=@A'1?:9@I'1?:9@aJ??%X?i?W?<6???Unknown
VHostSum"Sum_2(1?n??*7@9?n??*7@A?n??*7@I?n??*7@aX?.XV?iEoʘCA???Unknown
uHostReadVariableOp"div_no_nan/ReadVariableOp(1=
ףp?4@9=
ףp?4@A=
ףp?4@I=
ףp?4@a?????S?i???K???Unknown
?HostMul"7gradient_tape/MicroAudioDataModel/dropout/dropout/Mul_1(1??????2@9??????2@A??????2@I??????2@a?(S?4R?iD??T???Unknown
zHostMul")MicroAudioDataModel/dropout/dropout/Mul_1(1-????2@9-????2@A-????2@I-????2@a8?u???Q?iW}+?\???Unknown
gHostStridedSlice"strided_slice(1w??/?0@9w??/?0@Aw??/?0@Iw??/?0@ato??-P?iD<??e???Unknown
iHostWriteSummary"WriteSummary(1fffff&.@9fffff&.@Afffff&.@Ifffff&.@ax!?:$?L?i??3l???Unknown?
{ HostDataset"&Iterator::Model::MaxIntraOpParallelism(1;?O???D@9;?O???D@Ad;?O??+@Id;?O??+@a8O4{??J?i?????r???Unknown
?!HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1#??~j?*@9#??~j?*@A#??~j?*@I#??~j?*@arŁ??I?i?0!@y???Unknown
Z"HostArgMax"ArgMax(1;?O???*@9;?O???*@A;?O???*@I;?O???*@ax??޾?I?i??P????Unknown
?#HostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1)\???((@9)\???((@A)\???((@I)\???((@a?5G?i?fДi????Unknown
l$HostIteratorGetNext"IteratorGetNext(1Zd;?O'@9Zd;?O'@AZd;?O'@IZd;?O'@a?q\2F?i???+?????Unknown
?%HostBiasAddGrad";gradient_tape/MicroAudioDataModel/dense/BiasAdd/BiasAddGrad(1m????R&@9m????R&@Am????R&@Im????R&@a?>.??AE?i??F????Unknown
?&HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1Zd;?O?$@9Zd;?O?$@AZd;?O?$@IZd;?O?$@aiV???C?iux??*????Unknown
Y'HostPow"Adam/Pow(1?Zd?"@9?Zd?"@A?Zd?"@I?Zd?"@a??b^??A?is?????Unknown
?(HostReadVariableOp":MicroAudioDataModel/normalization/Reshape_1/ReadVariableOp(1?z?Ga"@9?z?Ga"@A?z?Ga"@I?z?Ga"@a?=U,?A?i?X$~ ????Unknown
?)HostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1???MbP"@9???MbP"@A???MbP"@I???MbP"@a֟b?pA?i?1??\????Unknown
t*HostAssignAddVariableOp"AssignAddVariableOp(1??ʡE"@9??ʡE"@A??ʡE"@I??ʡE"@a??^?eA?iK	???????Unknown
x+HostSqrt"&MicroAudioDataModel/normalization/Sqrt(1V-??/"@9V-??/"@AV-??/"@IV-??/"@aC?2??PA?i??2
????Unknown
x,HostMul"'MicroAudioDataModel/dropout/dropout/Mul(1H?z?"@9H?z?"@AH?z?"@IH?z?"@a8? ?*A?i1֍?T????Unknown
?-HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1V-??!@9V-??!@AV-??!@IV-??!@aJJ8??@?i??f?????Unknown
v.HostAssignAddVariableOp"AssignAddVariableOp_2(1!?rh?m!@9!?rh?m!@A!?rh?m!@I!?rh?m!@a?+???@?i??m?????Unknown
[/HostAddV2"Adam/add(1F?????@9F?????@AF?????@IF?????@a??ה?}=?iƶ\????Unknown
`0HostDivNoNan"
div_no_nan(1)\???(@9)\???(@A)\???(@I)\???(@a<?ː??;?iB?y?Ծ???Unknown
?1HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1?I+@9?I+@A?I+@I?I+@a?????:?i?:?*????Unknown
?2HostReadVariableOp"8MicroAudioDataModel/normalization/Reshape/ReadVariableOp(1/?$?@9/?$?@A/?$?@I/?$?@a?>?,0:?i@???p????Unknown
?3HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1???(\@9???(\@A???(\@I???(\@a????p?7?i???(l????Unknown
w4HostReadVariableOp"div_no_nan_1/ReadVariableOp(1?n???@9?n???@A?n???@I?n???@a?ĉ?\?6?i?BnA????Unknown
o5HostReadVariableOp"Adam/ReadVariableOp(1?? ?r?@9?? ?r?@A?? ?r?@I?? ?r?@a????5?iх??????Unknown
w6HostReadVariableOp"div_no_nan/ReadVariableOp_1(1??n?@@9??n?@@A??n?@@I??n?@@aryczH3?i?? d????Unknown
v7HostAssignAddVariableOp"AssignAddVariableOp_3(1?5^?I@9?5^?I@A?5^?I@I?5^?I@a?[զ?3?i??!??????Unknown
V8HostCast"Cast(1???Q?@9???Q?@A???Q?@I???Q?@a.K;??2?i9??????Unknown
~9HostMaximum")MicroAudioDataModel/normalization/Maximum(1???Q?@9???Q?@A???Q?@I???Q?@a.K;??2?il?p?x????Unknown
X:HostEqual"Equal(1??x?&?@9??x?&?@A??x?&?@I??x?&?@a??H??2?i???????Unknown
?;HostReadVariableOp"0MicroAudioDataModel/dense/BiasAdd/ReadVariableOp(1?O??n?@9?O??n?@A?O??n?@I?O??n?@a?ҫ2?i???%????Unknown
?<HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1?$??C@9?$??C@A?$??C@I?$??C@a?򐆦W2?i???o????Unknown
v=HostAssignAddVariableOp"AssignAddVariableOp_4(11?Z?@91?Z?@A1?Z?@I1?Z?@a?3?y??1?i??????Unknown
?>HostDataset"AIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl(1P??n?@9P??n?@AP??n?@IP??n?@ay?8Y?1?i???????Unknown
b?HostDivNoNan"div_no_nan_1(1?Zd;_@9?Zd;_@A?Zd;_@I?Zd;_@a?A? 9~1?it:h????Unknown
e@Host
LogicalAnd"
LogicalAnd(1?G?z@9?G?z@A?G?z@I?G?z@aEY?!71?i_F?I:????Unknown?
?AHostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1??Q??@9??Q??@A??Q??@I??Q??@a?{f???0?i.s6P????Unknown
vBHostAssignAddVariableOp"AssignAddVariableOp_1(1?E???T@9?E???T@A?E???T@I?E???T@as]?m??0?i:0D2`????Unknown
[CHostPow"
Adam/Pow_1(1?t??@9?t??@A?t??@I?t??@a?^O??0?i&zz,a????Unknown
?DHostReadVariableOp"/MicroAudioDataModel/dense/MatMul/ReadVariableOp(1333333@9333333@A333333@I333333@a??oCT?,?i"??A-????Unknown
dEHostDataset"Iterator::Model(1?(\?µF@9?(\?µF@A??? ?r@I??? ?r@a?Mw?
,?i?.??????Unknown
?FHostReadVariableOp"0MicroAudioDataModel/conv2d/Conv2D/ReadVariableOp(1+??@9+??@A+??@I+??@a&?g??w+?i??^?????Unknown
XGHostCast"Cast_2(1??(\??@9??(\??@A??(\??@I??(\??@aǹQ??n*?i+??GL????Unknown
~HHostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1^?I+@9^?I+@A^?I+@I^?I+@a???mE?)?iz?,?????Unknown
TIHostMul"Mul(1??ʡE?
@9??ʡE?
@A??ʡE?
@I??ʡE?
@a???o)?i(c?????Unknown
xJHostReadVariableOp"Adam/Identity/ReadVariableOp(1??ʡE?@9??ʡE?@A??ʡE?@I??ʡE?@aw?
?%?i???????Unknown
XKHostCast"Cast_3(1q=
ףp@9q=
ףp@Aq=
ףp@Iq=
ףp@a???bj$?i`/??!????Unknown
XLHostCast"Cast_4(1?Zd;?@9?Zd;?@A?Zd;?@I?Zd;?@aM?3"??"?i?RL~P????Unknown
yMHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1???x?&@9???x?&@A???x?&@I???x?&@a?,??Y<"?i??Ct????Unknown
?NHostReadVariableOp"1MicroAudioDataModel/conv2d/BiasAdd/ReadVariableOp(1!?rh??@9!?rh??@A!?rh??@I!?rh??@a2?s??"?iH"埔????Unknown
[OHostCast"	Adam/Cast(1d;?O????9d;?O????Ad;?O????Id;?O????a??S? ?iؾ??}????Unknown
zPHostReadVariableOp"Adam/Identity_1/ReadVariableOp(1??Q???9??Q???A??Q???I??Q???a?????i??v?<????Unknown
zQHostReadVariableOp"Adam/Identity_2/ReadVariableOp(1/?$???9/?$???A/?$???I/?$???a?0|e??i?+,^?????Unknown
aRHostIdentity"Identity(1V-???9V-???AV-???IV-???a??t??i     ???Unknown?*?O
?HostConv2DBackpropFilter"Dgradient_tape/MicroAudioDataModel/conv2d/Conv2D/Conv2DBackpropFilter(1ףp=
ǝ@9ףp=
ǝ@Aףp=
ǝ@Iףp=
ǝ@aI?r"??iI?r"???Unknown
zHost_FusedConv2D" MicroAudioDataModel/conv2d/Relu6(1??x馂?@9??x馂?@A??x馂?@I??x馂?@a??B??7??i??m????Unknown
?Host	Relu6Grad"8gradient_tape/MicroAudioDataModel/conv2d/Relu6/Relu6Grad(1?l?????@9?l?????@A?l?????@I?l?????@a??O3??iY|?/????Unknown
~HostMaxPool")MicroAudioDataModel/max_pooling2d/MaxPool(1-??????@9-??????@A-??????@I-??????@a8??ϵ?i?t-፿???Unknown
?HostBiasAddGrad"<gradient_tape/MicroAudioDataModel/conv2d/BiasAdd/BiasAddGrad(1)\???@s@9)\???@s@A)\???@s@I)\???@s@a??h????i,t??vK???Unknown
?HostMaxPoolGrad"Cgradient_tape/MicroAudioDataModel/max_pooling2d/MaxPool/MaxPoolGrad(1??n?@s@9??n?@s@A??n?@s@I??n?@s@a?"Y????iW?V????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_2/ResourceApplyAdam(1??S???p@9??S???p@A??S???p@I??S???p@a??@?徥?if?uD3???Unknown
?HostRandomUniform"@MicroAudioDataModel/dropout/dropout/random_uniform/RandomUniform(1
ףp=?e@9
ףp=?e@A
ףp=?e@I
ףp=?e@a?3?v?қ?i??)????Unknown
?	HostMatMul"0gradient_tape/MicroAudioDataModel/dense/MatMul_1(1bX9??b@9bX9??b@AbX9??b@IbX9??b@a?*B????i\???????Unknown
z
HostCast"(MicroAudioDataModel/dropout/dropout/Cast(1^?I?T@9^?I?T@A^?I?T@I^?I?T@a??D?p??i?????<???Unknown
~HostRealDiv")MicroAudioDataModel/normalization/truediv(17?A`??S@97?A`??S@A7?A`??S@I7?A`??S@a?k????i?6%?נ???Unknown
?HostMatMul".gradient_tape/MicroAudioDataModel/dense/MatMul(1X9??v^L@9X9??v^L@AX9??v^L@IX9??v^L@a???˾:??i^T??????Unknown
vHostSub"%MicroAudioDataModel/normalization/sub(1?A`?ЂK@9?A`?ЂK@A?A`?ЂK@I?A`?ЂK@a??kZ????i??7y0???Unknown
?HostResourceApplyAdam""Adam/Adam/update/ResourceApplyAdam(1?|?5^JI@9?|?5^JI@A?|?5^JI@I?|?5^JI@a+?s?S@??iM?ǆzq???Unknown
?HostGreaterEqual"0MicroAudioDataModel/dropout/dropout/GreaterEqual(1?p=
?F@9?p=
?F@A?p=
?F@I?p=
?F@acZ?(K|?i<?????Unknown
{Host_FusedMatMul"!MicroAudioDataModel/dense/BiasAdd(1ףp=
?E@9ףp=
?E@Aףp=
?E@Iףp=
?E@a{f??-?{?i?HT??????Unknown
?HostMul"Ugradient_tape/sparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/mul(1????K??@9????K??@A????K??@I????K??@a??6u?t?i???Σ
???Unknown
uHostFlushSummaryWriter"FlushSummaryWriter(1?????+?@9?????+?@A?????+?@I?????+?@a?߳??t?i?*?2???Unknown?
?Host#SparseSoftmaxCrossEntropyWithLogits"gsparse_categorical_crossentropy/SparseSoftmaxCrossEntropyWithLogits/SparseSoftmaxCrossEntropyWithLogits(1o??ʁ<@9o??ʁ<@Ao??ʁ<@Io??ʁ<@a?4?NrQr?i뺭VW???Unknown
?HostDataset"0Iterator::Model::MaxIntraOpParallelism::Prefetch(1? ?rh?;@9? ?rh?;@A? ?rh?;@I? ?rh?;@a??~??q?i]?fD{???Unknown
?HostResourceApplyAdam"$Adam/Adam/update_1/ResourceApplyAdam(1?z?G?:@9?z?G?:@A?z?G?:@I?z?G?:@a??m?Eq?i??@ߡ????Unknown
?HostResourceApplyAdam"$Adam/Adam/update_3/ResourceApplyAdam(1?/?$?9@9?/?$?9@A?/?$?9@I?/?$?9@a?U(m?p?if????????Unknown
?HostDataset"=Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCache(1{?G??=@9{?G??=@A'1?:9@I'1?:9@atZ5??5p?i??|V????Unknown
VHostSum"Sum_2(1?n??*7@9?n??*7@A?n??*7@I?n??*7@a?%i#?m?iAe??????Unknown
uHostReadVariableOp"div_no_nan/ReadVariableOp(1=
ףp?4@9=
ףp?4@A=
ףp?4@I=
ףp?4@aUSmf~j?i??`?????Unknown
?HostMul"7gradient_tape/MicroAudioDataModel/dropout/dropout/Mul_1(1??????2@9??????2@A??????2@I??????2@a?z???bh?is??/???Unknown
zHostMul")MicroAudioDataModel/dropout/dropout/Mul_1(1-????2@9-????2@A-????2@I-????2@a?????g?i???G???Unknown
gHostStridedSlice"strided_slice(1w??/?0@9w??/?0@Aw??/?0@Iw??/?0@at?~Eo?e?i9
Q??]???Unknown
iHostWriteSummary"WriteSummary(1fffff&.@9fffff&.@Afffff&.@Ifffff&.@a?4???_c?in?M?q???Unknown?
{HostDataset"&Iterator::Model::MaxIntraOpParallelism(1;?O???D@9;?O???D@Ad;?O??+@Id;?O??+@aZP??a?i?ݴ?????Unknown
?HostSum"1sparse_categorical_crossentropy/weighted_loss/Sum(1#??~j?*@9#??~j?*@A#??~j?*@I#??~j?*@a?6JW=Wa?i?]4?>????Unknown
Z HostArgMax"ArgMax(1;?O???*@9;?O???*@A;?O???*@I;?O???*@a??iE[Na?i??yM?????Unknown
?!HostDivNoNan"Lgradient_tape/sparse_categorical_crossentropy/weighted_loss/value/div_no_nan(1)\???((@9)\???((@A)\???((@I)\???((@aX?_?_?id?)?????Unknown
l"HostIteratorGetNext"IteratorGetNext(1Zd;?O'@9Zd;?O'@AZd;?O'@IZd;?O'@a?KM???]?i
???????Unknown
?#HostBiasAddGrad";gradient_tape/MicroAudioDataModel/dense/BiasAdd/BiasAddGrad(1m????R&@9m????R&@Am????R&@Im????R&@aD?🪰\?iU???f????Unknown
?$HostTile"Bgradient_tape/sparse_categorical_crossentropy/weighted_loss/Tile_1(1Zd;?O?$@9Zd;?O?$@AZd;?O?$@IZd;?O?$@a?9?C?iZ?i?x???????Unknown
Y%HostPow"Adam/Pow(1?Zd?"@9?Zd?"@A?Zd?"@I?Zd?"@a???X?i???"?????Unknown
?&HostReadVariableOp":MicroAudioDataModel/normalization/Reshape_1/ReadVariableOp(1?z?Ga"@9?z?Ga"@A?z?Ga"@I?z?Ga"@a4%???W?i?W??t????Unknown
?'HostCast"?sparse_categorical_crossentropy/weighted_loss/num_elements/Cast(1???MbP"@9???MbP"@A???MbP"@I???MbP"@a9??$]?W?i7?[9???Unknown
t(HostAssignAddVariableOp"AssignAddVariableOp(1??ʡE"@9??ʡE"@A??ʡE"@I??ʡE"@a$?d??{W?i?N? ????Unknown
x)HostSqrt"&MicroAudioDataModel/normalization/Sqrt(1V-??/"@9V-??/"@AV-??/"@IV-??/"@a??^@_W?i?O ?????Unknown
x*HostMul"'MicroAudioDataModel/dropout/dropout/Mul(1H?z?"@9H?z?"@AH?z?"@IH?z?"@a??L?+W?i\WƷ<&???Unknown
?+HostStridedSlice"-sparse_categorical_crossentropy/strided_slice(1V-??!@9V-??!@AV-??!@IV-??!@a?z?9??V?i?Q㓎1???Unknown
v,HostAssignAddVariableOp"AssignAddVariableOp_2(1!?rh?m!@9!?rh?m!@A!?rh?m!@I!?rh?m!@a6_??eV?i4?ނ?<???Unknown
[-HostAddV2"Adam/add(1F?????@9F?????@AF?????@IF?????@a'xab??S?i?????F???Unknown
`.HostDivNoNan"
div_no_nan(1)\???(@9)\???(@A)\???(@I)\???(@a??2?ݼR?iWKjjP???Unknown
?/HostCast"`sparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast(1?I+@9?I+@A?I+@I?I+@a?Xw?R?i????Y???Unknown
?0HostReadVariableOp"8MicroAudioDataModel/normalization/Reshape/ReadVariableOp(1/?$?@9/?$?@A/?$?@I/?$?@aj???s?Q?i?vn??a???Unknown
?1HostPack"/sparse_categorical_crossentropy/Reshape_1/shape(1???(\@9???(\@A???(\@I???(\@a????hP?i?P?+?i???Unknown
w2HostReadVariableOp"div_no_nan_1/ReadVariableOp(1?n???@9?n???@A?n???@I?n???@a??Ĥ??N?i???q???Unknown
o3HostReadVariableOp"Adam/ReadVariableOp(1?? ?r?@9?? ?r?@A?? ?r?@I?? ?r?@a?`???pM?i;?~H?x???Unknown
w4HostReadVariableOp"div_no_nan/ReadVariableOp_1(1??n?@@9??n?@@A??n?@@I??n?@@a?oR??J?i׃?{???Unknown
v5HostAssignAddVariableOp"AssignAddVariableOp_3(1?5^?I@9?5^?I@A?5^?I@I?5^?I@a?`????I?i??p??????Unknown
V6HostCast"Cast(1???Q?@9???Q?@A???Q?@I???Q?@aJ549?WI?i?K??A????Unknown
~7HostMaximum")MicroAudioDataModel/normalization/Maximum(1???Q?@9???Q?@A???Q?@I???Q?@aJ549?WI?i	???????Unknown
X8HostEqual"Equal(1??x?&?@9??x?&?@A??x?&?@I??x?&?@a?????NI?i?
ɤ?????Unknown
?9HostReadVariableOp"0MicroAudioDataModel/dense/BiasAdd/ReadVariableOp(1?O??n?@9?O??n?@A?O??n?@I?O??n?@ac????'I?i???t5????Unknown
?:HostDivNoNan"3sparse_categorical_crossentropy/weighted_loss/value(1?$??C@9?$??C@A?$??C@I?$??C@a"?????H?igrs?e????Unknown
v;HostAssignAddVariableOp"AssignAddVariableOp_4(11?Z?@91?Z?@A1?Z?@I1?Z?@a#c??GH?i@?8?w????Unknown
?<HostDataset"AIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl(1P??n?@9P??n?@AP??n?@IP??n?@a,?????G?ij??qj????Unknown
b=HostDivNoNan"div_no_nan_1(1?Zd;_@9?Zd;_@A?Zd;_@I?Zd;_@a=?}.r?G?i?9.?Q????Unknown
e>Host
LogicalAnd"
LogicalAnd(1?G?z@9?G?z@A?G?z@I?G?z@a?Z??`<G?i??P? ????Unknown?
??HostCast"bsparse_categorical_crossentropy/ArithmeticOptimizer/ReorderCastLikeAndValuePreserving_int64_Cast_1(1??Q??@9??Q??@A??Q??@I??Q??@a.??"?F?i?????????Unknown
v@HostAssignAddVariableOp"AssignAddVariableOp_1(1?E???T@9?E???T@A?E???T@I?E???T@aqzzaHFF?i@CAS????Unknown
[AHostPow"
Adam/Pow_1(1?t??@9?t??@A?t??@I?t??@ar2??E?i?OJ?????Unknown
?BHostReadVariableOp"/MicroAudioDataModel/dense/MatMul/ReadVariableOp(1333333@9333333@A333333@I333333@awp???gC?i?<??????Unknown
dCHostDataset"Iterator::Model(1?(\?µF@9?(\?µF@A??? ?r@I??? ?r@a???>?B?iX_vQ????Unknown
?DHostReadVariableOp"0MicroAudioDataModel/conv2d/Conv2D/ReadVariableOp(1+??@9+??@A+??@I+??@a?????B?i/v?????Unknown
XEHostCast"Cast_2(1??(\??@9??(\??@A??(\??@I??(\??@a?J?W??A?i25i????Unknown
~FHostAssignAddVariableOp"Adam/Adam/AssignAddVariableOp(1^?I+@9^?I+@A^?I+@I^?I+@a?m?-uA?i?+?e?????Unknown
TGHostMul"Mul(1??ʡE?
@9??ʡE?
@A??ʡE?
@I??ʡE?
@a?uΊ**A?i<?J?????Unknown
xHHostReadVariableOp"Adam/Identity/ReadVariableOp(1??ʡE?@9??ʡE?@A??ʡE?@I??ʡE?@a2???Q0=?i3???????Unknown
XIHostCast"Cast_3(1q=
ףp@9q=
ףp@Aq=
ףp@Iq=
ףp@aW??Ӎ;?iS??(????Unknown
XJHostCast"Cast_4(1?Zd;?@9?Zd;?@A?Zd;?@I?Zd;?@a???d??9?i?Po?Y????Unknown
yKHostReadVariableOp"div_no_nan_1/ReadVariableOp_1(1???x?&@9???x?&@A???x?&@I???x?&@a?4@??8?i?X??m????Unknown
?LHostReadVariableOp"1MicroAudioDataModel/conv2d/BiasAdd/ReadVariableOp(1!?rh??@9!?rh??@A!?rh??@I!?rh??@a~)?,_S8?iu??x????Unknown
[MHostCast"	Adam/Cast(1d;?O????9d;?O????Ad;?O????Id;?O????a,??n?3?i?i[?????Unknown
zNHostReadVariableOp"Adam/Identity_1/ReadVariableOp(1??Q???9??Q???A??Q???I??Q???a;7;?G$0?iqN??????Unknown
zOHostReadVariableOp"Adam/Identity_2/ReadVariableOp(1/?$???9/?$???A/?$???I/?$???a_H?_t?"?i?p?? ????Unknown
aPHostIdentity"Identity(1V-???9V-???AV-???IV-???aN??q??i     ???Unknown?2Nvidia GPU (Maxwell)