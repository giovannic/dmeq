import jax

# The search this package feeds runs at f64: a novel curve family at f32 through
# two sequential scan recursions produces NaNs. Test there too.
jax.config.update('jax_enable_x64', True)
