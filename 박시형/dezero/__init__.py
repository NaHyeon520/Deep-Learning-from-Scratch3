is_simple_core = False

if is_simple_core:
  from dezero.core_simple import (
    Variable,
    Function,
    using_config,
    no_grad,
    as_array,
    as_variable,
    setup_variable
  )
else:
  from dezero.core import (
    Variable,
    Parameter,
    Function,
    using_config,
    no_grad,
    as_array,
    as_variable,
    setup_variable
  )

setup_variable()