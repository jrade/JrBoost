#  Copyright 2022 Johan Rade <johan.rade@gmail.com>.
#  Distributed under the MIT license.
#  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

try:
    from ._jrboost import *
except:
    from _jrboost import * 

from ._util import *
