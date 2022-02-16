export const urls = {
  registerUrl: "http://localhost:5002/api/auth/register",
  loginUrl: "http://localhost:5002/api/auth/login",
  authenticatedUrl: "http://localhost:5002/api/auth/authenticated",
  logoutUrl: "http://localhost:5002/api/auth/logout",
  environmentAddressesUrl: "http://localhost:5002/backend/api/environment",
  environmentsDataDistributionUrl:
    "http://localhost:5002/backend/api/dataset/distribution",
  environmentsTrainingDistributionUrl:
    "http://localhost:5002/backend/api/dataset/training/distribution",
  environmentCreateUrl: "http://localhost:5002/backend/api/environment/create",
  gatewayBackendUrl: "http://localhost:5002/backend",
  environmentDeleteUrl: "http://localhost:5002/backend/api/environment/delete",
  environmentTrainingDistributionAddUrl:
    "http://localhost:5002/backend/api/environment/dataset/distribution",
  environmentDataDistributionTrainAddUrl:
    "http://localhost:5002/backend/api/environment/dataset/data",
  environmentDataDistributionValidationAddUrl:
    "http://localhost:5002/backend/api/environment/dataset/validation",
  environmentDataDistributionTestAddUrl:
    "http://localhost:5002/backend/api/environment/dataset/test",
  environmentModelAddUrl: "http://localhost:5002/backend/api/environment/model",
};
