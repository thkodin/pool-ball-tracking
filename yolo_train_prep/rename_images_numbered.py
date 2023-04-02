# IMPORTANT
# =========
# Renames a given directory files to 1.jpg, 1.txt, 2.jpg, 2.txt, 3.jpg, 3.txt...
# BEST TO BE SAFE. We recommend backing up your original folder as the rename operation will remove the files from the old directory.
# YOLO expects data in pairs of image, text file containing detection information in that image. We assume the user already has the pairs and that the filename of both
# the image and corresponding text file is the same (as is the case if you use labelImg). However, the numbering or naming is random and wonky and the user would like
# a more pleasing, numbered outlook on this data. YOLO also requires a classes.txt which contains the class names, and labelImg stores this in the same directory as the
# .txt files, so make sure to remove it (or its equivalent for your software) before executing this script. Otherwise, you'd get an error!

# Finally, please make sure to verify that the correct bounding boxes are preserved with your labelling software (we tested with labelImg)
import os
import re 

imgDir = os.path.abspath("D:/final")            # path to old directory with file names you want to change
new_imgDir = os.path.abspath("D:/finalfinal")   # path to new directory where the file renames will be saved

name = 1                # EDITABLE*** start image number - can be offset if already having numbered images upto a point e.g. start renaming to 200 and onwards
itr = 0                 # keep track of files and make some checks on validity of pair as well as update names
bad_file = False        # if anything other than a .jpg or .txt file is encountered
no_pair_txt = False     # if an un-paired .jpg found (no .txt with same name)
no_pair_jpg = False     # if an un-paired .txt found (no .jpg with same name)

# os.listdir() arranges arbitrarily (filesystem dependent). To preserve the order of files, we ensure that the list is alphanumerically sorted ourselves. Since "j" comes
# before "t", <file_name>.txt will always come AFTER <file_name>.jpg for the same file_name. Thus, when checking for .jpg - .txt pairs, a .txt file must be checked with
# the file before it (must be .jpg) in the list provided by os.listdir(), and a .jpg file must be checked with the file after it (must be .txt).

# The following (storing listdir() results) is necessary as .rename() removes files from original directory once renamed. Thus if you were to check for pairs without 
# storing an image of the original folder, os.listdir() would be dynamically changing every iteration, resulting in constantly failed checks as the .jpg in the i'th
# iteration would no longer be listed (having been deleted by rename()) in the i+1'th iteration for the corresponding .txt file. A static copy of the original directory
# files is thus absolutely necessary.
flistUnsorted = os.listdir(imgDir)

# Sort alphanumerically. Essentially re.split separates the text and numerics into separate list elements in order of apperance left to right. Thus, you get a list of 
# strings. Then, the list comprehension iterates over these elements and decides whether to set the comparison as integers or strings depending on whether the string
# element represents numerics or alphabetical text (.isdigit() method). Once all elements of this list are exhausted, the resultant list is a list of mixed integers
# and strings! This is the true alphanumeric order we require, so we set it as the key for our sorted() function. This separation of alphabets and numerics is necessary 
# because str(1101) is 'shorter' than str(121)!

# ***QUICK DEMONSTRATION***

# import re
# ke = lambda text: [int(t) if t.isdigit() else t for t in re.split('([0-9]+)', text)]
# name1 = "uer10938keu.jpg"
# name2 = "109uerke38f.jpg"   # note how the first element in this one will be a "" (empty character string) in print(ke(name2)). "" is always < "non empty string", and 
#                             # the sort will therefore put the name starting with an integer before a name starting with a non-integer! This neat little trick of padding
#                             # an empty character for names starting with digits ensures that the illegal comparison of strings and integers NEVER takes place because
#                             # the first character in the list returned by 'ke' is ALWAYS a string character! Any future splits would only occur once a digit appears
#                             # and so on. This is a very beautiful method proposed in: https://stackoverflow.com/questions/2669059/how-to-sort-alpha-numeric-set-in-python 
# print(ke(name1))
# print(ke(name2))
# lst = [name1, name2]
# print(lst)
# print(sorted(lst, key = ke))

# lst = ['10', '20', '100']       # sorted list with string integers
# print(lst)
# print(sorted(lst))              # sorted() has now unsorted our list! str(100) < str(20)! Must give key = int for this to work, but this will mess up alphanumerics.
# print(sorted(lst, key = ke))    # preserves order properly because all digit strings are treated as int.

# lst = ['10def32', 'def20abc', '100abc']   # sorted alphanumeric list; integers have lowest value when arranging in order and so will always come first in ascending order
#                                           # thus, 10def < 20abc, but 10def > 10abc. Also, '<any_integer><text>' is ALWAYS shorter than '<text><any_integer>'!  
# print(lst)
# badlst = sorted(lst)                   # sorted() has, once again, unsorted the list! But int("<text>") isn't valid in python, so now what?
# print(badlst)
# print(sorted(badlst, key = ke))        # properly sorts the messed up results

# ***END DEMONSTRATION***

fileList = sorted(flistUnsorted, key = lambda text: [int(t) if t.isdigit() else t for t in re.split('([0-9]+)', text)])
print("Found the following...\n", fileList)

for f in fileList:  # EDITABLE*** this gives a list of filenames ALPHANUMERICALLY ARRANGED. If you had a mix of numbered and un-numbered data, you should 
                    # change this to start accordingly. E.g. if you had 1.jpg to 200.jpg numbered images (200 imgs) with paired .txts (total 400 numbered 
                    # files), and after this it is 1600541, 1674030 kind of thing, do os.listdir(imgDir)[400:] to start renaming from the 401st file (201st 
                    # jpg-txt pair) inclusive and onwards. Change variable name accordingly (in this case name should start from 201).
                     
    # os.rename() expects 'oldPath + filename, newPath + filename' kind of arguments.

    # 1. Found a .jpg image. [-3:] gets the last 3 characters from the string (extension of file)
    if f[-3:] == "jpg":
        # 1.1. CHECK FOR PAIRED TXT.
        # The first condition before 'and' checks the name of the NEXT file. The 2nd condition checks extension name.
        if fileList[itr + 1].split(".", 1)[0] == f.split(".", 1)[0] and fileList[itr + 1][-3:] == "txt":  
#          ^                                     ^                      ^
#          |                                     |                      |
#          Get next file name before "."         |                      Get extension of next file (must be "txt" to proceed)
#                                                |                           
#                                                Get current file name before "."

            # If paired .txt file available, rename image file and update iteration count.
            os.rename(os.path.join(imgDir, f), os.path.join(new_imgDir, str(name) + ".jpg"))
            itr += 1
        # 1.2. Failed pair test; exit and print error.
        else:
            no_pair_txt = True
            break

    # 2. Found a .txt file.
    elif f[-3:] == "txt":
        # 2.1. CHECK FOR PAIRED JPG (mirror condition of 1.1.)
        # The first condition before 'and' checks the name of the PREVIOUS file. The 2nd condition checks extension name.
        if fileList[itr - 1].split(".", 1)[0] == f.split(".", 1)[0] and fileList[itr - 1][-3:] == "jpg":
            # If paired .jpg file available, rename image file and update iteration count.
            os.rename(os.path.join(imgDir, f), os.path.join(new_imgDir, str(name) + ".txt"))
            itr += 1
        # 2.2. Failed pair test; exit and print error.
        else:
            no_pair_jpg = True
            break
    
    # 3. Unable to find either of one extensions; exit and print error.
    else:
        bad_file = True
        break

    # 4. itr = 0 to 1 (a .jpg or .txt file found) --> itr = 1 to 2 (paired file for previous step found) --> UPDATE NAME (% 2, every 2 steps) ...
    # ... --> itr = 2 to 3 (a new .jpg or .txt found) --> itr = 3 to 4 (paired file found) --> UPDATE NAME ... until all files exhausted or error.
    if itr % 2 == 0 and itr != 0:
        name += 1

if no_pair_txt is True:
    print("[ERROR] Encountered .jpg without a paired .txt; exiting...\nPlease ensure all files are available as jpg - txt pairs.")
elif no_pair_jpg is True:
    print("[ERROR] Encountered .txt without a paired .jpg; exiting...\nPlease ensure all files are available as jpg - txt pairs.")
elif bad_file is True:
    print("[ERROR] Encountered a file that was neither .jpg nor .txt; exiting...\nPlease ensure that only .jpg and .txt files are present in the given directory.")
else:
    print("[INFO] Successfully renamed all files.")
