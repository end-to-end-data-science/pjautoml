ls -fld --color $((ls -dfl $(git ls-tree master --name-only) | grep "^d"; ls -dfl $(git ls-tree master --name-only) | grep "^-"; ls -dfl $(git ls-tree master --name-only) | grep "^l") | tr -s " " | cut -d" " -f9)

