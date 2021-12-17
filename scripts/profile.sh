# Runs Python profiling
# Results are saved to profile.out as well as printed to standard output
# Usage profile.sh <script>.py <args>
#    OR profile.sh <module>.<entry_point> <args>

# Run profiling and save results to disk
python -m cProfile -o profile.out $@

# Parse results to a text file
echo -e 'sort cumtime\nstats' | python -m pstats profile.out > profile.txt

# Print some of the text file
head -n 64 profile.txt
