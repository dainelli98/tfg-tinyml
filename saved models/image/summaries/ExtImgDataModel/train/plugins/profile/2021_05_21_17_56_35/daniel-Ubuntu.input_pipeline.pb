  *	G?zn??@2?
jIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[0]::ParallelMapV2 T???
? @!Z?VB?8O@)T???
? @1Z?VB?8O@:Preprocessing2?
JIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2ڐf?	@!>?????W@)?ƠB??1?ӊ?4?6@:Preprocessing2?
XIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip ??0|d@!?Dh?QQ@)?س?2??1?:?N"?@:Preprocessing2?
SIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2::Shuffle -_???=@!D9???R@)A?v??1??>???	@:Preprocessing2?
wIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[0]::ParallelMapV2::TensorSlice ?]??Nw??!J?$?'@)?]??Nw??1J?$?'@:Preprocessing2?
hIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImpl::BatchV2::Shuffle::Zip[1]::TensorSlice ?Ss??P??!????|??)?Ss??P??1????|??:Preprocessing2g
0Iterator::Model::MaxIntraOpParallelism::Prefetch?;??.??!??c?{??)?;??.??1??c?{??:Preprocessing2]
&Iterator::Model::MaxIntraOpParallelism?%r????!????
<??)??c[???1?~,Y??:Preprocessing2x
AIterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCacheImplh]??@O	@!E?wl?W@)???#0??1??%)}??:Preprocessing2F
Iterator::Model\<???r??!s???????)?2p@Kw?1?\?,???:Preprocessing2t
=Iterator::Model::MaxIntraOpParallelism::Prefetch::MemoryCache5`??iU	@!?i?m??W@)?4?;?h?1?*j?4??:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysisk
unknownTNo step time measured. Therefore we cannot tell where the performance bottleneck is.no*noZno#You may skip the rest of this page.BZ
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown
  " * 2 : B J R Z b JGPUb??No step marker observed and hence the step time is unknown. This may happen if (1) training steps are not instrumented (e.g., if you are not using Keras) or (2) the profiling duration is shorter than the step time. For (1), you need to add step instrumentation; for (2), you may try to profile longer.