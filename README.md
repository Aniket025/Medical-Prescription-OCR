# Opensoft
IIT Khargpur -RP Hall Opensoft Master repository

Objective :
The objective here is to let allow a doctor to write his prescriptions the conventional way (i.e., using their pen and paper). From the scanned version of the prescription, a handwritten character recognition will be followed to capture the data (name of the patient, symptoms, findings, prescription of medicine, tests, advice, etc.) written by the doctor. Since,   the accuracy rate of the state-of-the-art hand written character reorganization is not still up to the acceptable level, we propose to apply an error correction mechanism to reduce the errors. The solution does not oppose the age-old convention and affordable as it is mostly a software solution with a minimum hardware requirement.
Working :
Input: A scan copy of a doctor’s handwritten prescription and stored as an image file.
Procedure (also, see the figure in the next page):
Model-1 : Image pre-processing: To deal with the low quality image, noise in scan, binarization, alignment, etc. This step will produce an acceptable and processable image form.

Model-2 : Image segmentation: In this step, we have to segment into a number of blocks identifying different regions such as computer printed parts, sketches, computer printed images, and hand-written texts, etc.  The blocks containing the hand-written texts are the regions of interest (RoIs). This step will return all the RoIs in the prescription document.  

Model-3 : Hand-written text recognition: The RoIs involving the doctor’s hand-written parts are the input in this step. For each such RoI, we have to extract words in them. Then, in each word, we have to identify the characters in it.  Thus, the output in this step will be the character images in each word in each RoI.

Model- 4 :  Character level recognition: Each character image in the last step, will be processed optically to recognize the character and finally it can be stored in the form of ASCII character. The outcome of this step is the ASCII form of each word in each RoI.

Model- 5 : Sloppy hand writing correction: The words are to be processed to check, if there is any spelling mistakes or predicting the correct words in the hand-written texts.  For this purpose, you should consult a “medicine vocabulary” (such a vocabulary available in the Net and freely downloadable).  Hint: You can follow any English language text prediction tool, and fine tune the same with a language resource model based on the “medicine vocabulary” as the corpus. Such a language prediction tool will predict the most probable correct words with their ranks. Output in this step will be the correct words in each RoI.

Output: The output is a document file which will place items as image in the place as it is they are there in the input prescription, except the digitized version of the doctor’s hand written text with corrections.



Model - 1 : Forked from https://github.com/Breta01/handwriting-ocr
