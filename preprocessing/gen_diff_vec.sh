RED='\033[0;31m'
NC='\033[0m'
BLUE='\033[0;34m'
YELLOW='\033[0;33m'

prompts_list='prompt.txt'
attributes_list='attr_adj.txt'
data="${1:-"cifar100"}"


echo "${NC}Generate text difference feats for ${BLUE}$data${NC}"

while read prompt; do
  while read attr_line; do
    attr="$(echo "$attr_line" | cut -d ' ' -f 1)"
    adj="$(echo "$attr_line" | cut -d ' ' -f 2)"
    echo "with [Prompt] ==> ${YELLOW}$prompt${NC}"
    echo "with [Attribute & Adjective] ==> ${RED}$attr & $adj${NC}"
    python3 gen_text_feats_clip.py --data "$data" --prompt "$prompt" --attr "$attr" --adj "$adj"
  done < $attributes_list
done < $prompts_list
