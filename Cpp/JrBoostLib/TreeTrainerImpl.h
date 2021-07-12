// Copyright (C) 2021 Johan Rade <johan.rade@gmail.com>
//  Copyright 2021 Johan Rade <johan.rade@gmail.com>.
//  Distributed under the MIT license.
//  (See accompanying file License.txt or copy at https://opensource.org/licenses/MIT)

#pragma once

#include "TreeTrainerImplA.h"
#include "TreeTrainerImplC.h"
#include "TreeTrainerImplD.h"

template<typename SampleIndex>
using TreeTrainerImpl = TreeTrainerImplD<SampleIndex>;
