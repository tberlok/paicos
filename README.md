python setup.py build_ext --inplace

python -u snap_loop_over_halos.py 246 > 246.log 2>&1 & disown

msg = 'python -u snap_loop_over_halos.py {} > {}.log 2>&1 & disown'
for ii in range(248):
   print(msg.format(ii, ii))
