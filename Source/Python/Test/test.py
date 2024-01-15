
import agaricus
import iris
import titanic

ok1 = agaricus.test()
ok2 = iris.test()
ok3 = titanic.test()

if ok1 and ok2 and ok3:
    print('ALL TESTS PASSED\n')
else:
    print('AT LEAST ONE TEST FAILED\n')
