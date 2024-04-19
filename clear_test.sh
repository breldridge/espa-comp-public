for testfile in $(find . -name "*TEST*")
do rm $testfile
done

for testfile in $(find . -name "*DAM*")
do rm $testfile
done

for testfile in $(find . -name "*RTM*")
do rm $testfile
done

for testfile in $(find . -name "*RHF*")
do rm $testfile
done

for testfile in $(find . -name "pre_start*")
do rm $testfile
done
