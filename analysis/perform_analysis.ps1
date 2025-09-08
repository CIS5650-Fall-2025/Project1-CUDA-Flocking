# If you're running this, make sure you do so from the repo root i.e. move this
# script out of the "analysis" folder! The program uses hardcoded relative paths
# to the "shaders" folder and will not find them otherwise.

$boids_counts = 10000, 25000, 50000, 100000, 250000, 750000, 1500000
$block_sizes = 64, 128, 256, 512, 1024

foreach ($boid_count in $boids_counts) {
    foreach ($block_size in $block_sizes) {
        .\build\bin\release\cis5650_boids.exe "$boid_count" "$block_size"
        Write-Host ""
    }
}
