provider "google" {
  project = "finalyearproject-338819"
  region  = "europe-west1"
  zone    = "europe-west1-b"
}

resource "google_compute_instance" "vm_instance" {
  name         = "terraform-test-instance"
  machine_type = "e2-micro"

  boot_disk {
    initialize_params {
      image = "debian-cloud/debian-10"
    }
  }
  network_interface {
    network = "default"
    access_config {}
  }

  metadata_startup_script = <<-EOF
          sudo apt-get update
          sudo apt-get install apt-transport-https ca-certificates curl gnupg2 software-properties-common
          curl -fsSL https://download.docker.com/linux/debian/gpg | sudo apt-key add -
          sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/debian buster stable"
          sudo apt-get update
          sudo apt-get install docker-ce docker-ce-cli containerd.io
          docker pull eu.gcr.io/finalyearproject-338819/fyp_main_controller
          docker run -p 5000:5000 eu.gcr.io/finalyearproject-338819/fyp_main_controller
        EOF

}
