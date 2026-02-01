PARENT="data/teams"

while IFS= read -r dir; do
    mkdir -p "$PARENT/$dir"
done < data/teams/teams_list.txt