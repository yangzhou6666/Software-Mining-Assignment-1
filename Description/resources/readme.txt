Should I follow this fault localization tool’s output?
Automated prediction of fault localization effectiveness

**********************
* Problem Definition *
**********************

Debugging is a crucial yet expensive activity to improve the reliability of software
systems. To reduce debugging cost, various fault localization tools have been proposed. A
spectrum-based fault localization tool often outputs an ordered list of program elements
sorted based on their likelihood to be the root cause of a set of failures (i.e., their suspiciousness scores). Despite the many studies on fault localization, unfortunately, however,
for many bugs, the root causes are often low in the ordered list. This potentially causes
developers to distrust fault localization tools. Recently, Parnin and Orso highlight in their
user study that many debuggers do not find fault localization useful if they do not find the
root cause early in the list. To alleviate the above issue, we build an oracle that could predict whether the output of a fault localization tool can be trusted or not. If the output is not
likely to be trusted, developers do not need to spend time going through the list of most
suspicious program elements one by one. Rather, other conventional means of debugging
could be performed. To construct the oracle, we extract the values of a number of features
that are potentially related to the effectiveness of fault localization. Building upon advances
in machine learning, we process these feature values to learn a discriminative model that is
able to predict the effectiveness of a fault localization tool output. In this work, we consider
an output of a fault localization tool to be effective if the root cause appears in the top 10
most suspicious program elements. We have evaluated our proposed oracle on 200 faulty
versions of Space, NanoXML, XML-Security, and the 7 programs in Siemens test suite.
Our experiments demonstrate that we could predict the effectiveness of 9 fault localization
tools with a precision, recall, and F-measure (harmonic mean of precision and recall) of up


File "features.dat" contains dataset for predicting effectiveness of fault localization tools.


***************
* LINE FORMAT *
***************
Each line is in the format

	<target> <feature>:<value> <feature>:<value> ... <feature>:<value>

with
- <target> denotes the target class of the data instance.
- <feature> denotes the feature number.
- <value> denotes the value of the corresponding feature.


*********
* CLASS *
*********
Each target class represents:
+1 -> effective
-1 -> ineffective


*******************
* FEATURE MAPPING *
*******************
Refer to Table 2 in the paper.

