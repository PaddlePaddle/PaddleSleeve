set -e

if [ $# != 1 ];then
    echo "ERROR!"
    echo "usage: sh run_test.sh PYTHON_BIN"
    exit 1
fi

PYTHON=$1

tests=$(ls ./ |grep test_)

for test in $tests
do
    echo begin test ${test}
    $PYTHON $test
done

echo "all tests pass"
set +e