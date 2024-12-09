#!/bin/bash
#
# Fetch results from remote cluster
#

arg0=$(basename "$0" .sh)
blnk=$(echo "$arg0" | sed 's/./ /g')

usage_info()
{
    echo "Usage: $arg0 [{-f|--from} user@host:/path/to/results] [{-t|--to} local/path/to/results]   \\"
    echo "       $blnk [{-s|--subset} results/subfolder] ...                                        \\"
    echo "       $blnk [{-p|--use-password}"
}

usage()
{
    exec 1>2   # Send standard output to standard error
    usage_info
    exit 1
}

error()
{
    echo "$arg0: $*" >&2
    exit 1
}

help()
{
    usage_info
    echo
    echo "  {--from} user@host:/path/to/results         -- Remote username, host, and path to fetch the results from"
    echo "  {--to} local/path/to/results                -- Local folder to store fetched results"
    echo "  {-s|--subset} results/subfolder             -- Limit the fetched results to the specified subset of results"
    echo "                                                   NB: Can be passed multiple times to expand the subset"
    echo "  {-p|--use-password}                         -- If provided, assume password is required for access on remote host"
    exit 0
}

parse_flags()
{

    RESULTS_SUBSETS=()
    SSH_USE_PASSWORD='!false!'

    while test $# -gt 0
    do
        case "$1" in
        (-f|--from)
            shift
            [ $# = 0 ] && error "Missing remote source of results: --from user@host:/path/to/results"
            REMOTE_RESULTS_DIR="$1"
            shift;;
        (-t|--to)
            shift
            [ $# = 0 ] && error "Missing local folder to store results: --to local/path/to/results"
            LOCAL_RESULTS_DIR="$1"
            shift;;
        (-s|--subset)
            shift
            [ $# = 0 ] && error "Missing subset of results to collect: --subset results/subfolder"
            RESULTS_SUBSETS+=("$1")
            shift;;
        (-p|--use-password)
            shift
            SSH_USE_PASSWORD="!true!"
            ;;
        (-h|--help)
            help;;
        (*) usage;;
        esac
    done

    if (( "${#RESULTS_SUBSETS[@]}" == 0)); then
      RESULTS_SUBSETS+=('!everything!')
    fi

}

fetch_results_passwordless()
{

  for csv_file in "${results_files[@]}"; do

    csv_parent_directory=$(dirname "${csv_file}")
    local_csv_parent_directory=${csv_parent_directory#"$remote_results_dir"}
    local_folder="${LOCAL_RESULTS_DIR}${local_csv_parent_directory}"

    mkdir -p "${local_folder}"
    echo "      --> Fetching ${local_folder}"
    scp "${ssh_endpoint}:${csv_file}" "${local_folder}" 2>/dev/null

  done

}

fetch_results_using_password()
{

  remote_tmp_zipped_results="${remote_results_dir}/__tmp_results.zip"
  local_tmp_zipped_results="${LOCAL_RESULTS_DIR}/__tmp_results.zip"

  ssh "${ssh_endpoint}" zip -r "${remote_tmp_zipped_results}" "${results_files[@]}"
  scp "${ssh_endpoint}":"${remote_tmp_zipped_results}" "${LOCAL_RESULTS_DIR}"

  unzip "${local_tmp_zipped_results}"

}

fetch_results()
{

  IFS=':' read -ra _split <<< "${REMOTE_RESULTS_DIR}"

  ssh_endpoint="${_split[0]}"
  remote_results_dir="${_split[1]}"

  for subset in "${RESULTS_SUBSETS[@]}"; do

    if [[ $subset == '!everything!' ]]; then
      echo '    --> Fetching all results'
      subset=""
    else
      echo "    --> Fetching results for subset: ${subset}"
    fi

    echo '[.] Discovering results file on remote host...'

    find_output=$(ssh "${ssh_endpoint}" find "${remote_results_dir}/${subset}" -type f -name 'progress.csv' 2>/dev/null)

    # find exit code 1: file not found
    if (( $? == 1)); then
      error "Remote results folder not found: ${remote_results_dir}"
    fi

    read -r -a results_files -d '\n' <<< "${find_output}"

    echo "[.] Found ${#results_files[@]} CSV result files to fetch"

    if [[ "${SSH_USE_PASSWORD}" == '!true!' ]]; then
      fetch_results_using_password
    else
      fetch_results_passwordless
    fi

  done

}

parse_flags "$@"

echo "[*] Fetching results from: $REMOTE_RESULTS_DIR"
echo "[*] Storing results in: $LOCAL_RESULTS_DIR"
echo "[*] Requested results subset: ${RESULTS_SUBSETS[*]}"
echo "[*] Use password auth: ${SSH_USE_PASSWORD}"

fetch_results