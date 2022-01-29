variable "nr_instances" {
  type        = number
  description = "The number of instances to create"
  default     = 1

  validation {
    condition     = var.nr_instances >= 1 && var.nr_instances <= 10
    error_message = "The instance number has to be between 1 and 10 (inclusive)."
  }
}

variable "user_id" {
  type        = string
  description = "The user name for the current environment"
}

variable "machine_type" {
  type        = string
  description = "The type of machine instance for the environment"
  default     = "e2-medium"
}

variable "instance_image" {
  type        = string
  description = "The type of image to use for the instance"
  default     = "projects/finalyearproject-338819/global/images/instance-image"
}
