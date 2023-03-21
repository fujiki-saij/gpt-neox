hostname=$1

ip_address=$(echo $hostname | cut -d '-' -f 2- | tr '-' '.')

echo $ip_address
