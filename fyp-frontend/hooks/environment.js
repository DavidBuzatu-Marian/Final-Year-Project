import { getConfig } from "../config/defaultConfig";
import useSWR from "swr";
import { fetcher } from "./user";

export const useEnvironment = () => {
  const {
    data,
    mutate,
    error: environmentsError,
  } = useSWR(getConfig()["environmentAddressesUrl"], fetcher, {
    onErrorRetry: (error, key, config, revalidate, { retryCount }) => {
      if (error.status === 404) return;

      if (retryCount >= 5) return;
    },
  });

  const loading = !data && !environmentsError;
  const environments =
    data && !environmentsError && Array.isArray(data) ? data : null;
  if (environments) {
    environments = environments.map((environment) => {
      return { ...environment, id: environment._id };
    });
  }
  return [environments, { mutate, loading }, environmentsError];
};

export const useEnvironmentTrainingLogs = (environmentId) => {
  const {
    data,
    mutate,
    error: trainingLogsError,
  } = useSWR(getConfig()["environmentTrainingLogsUrl"], fetcher, {
    onErrorRetry: (error, key, config, revalidate, { retryCount }) => {
      if (error.status === 404) return;

      if (retryCount >= 5) return;
    },
  });

  const loadingTrainingLogs = !data && !trainingLogsError;
  const trainingLogs =
    data && !trainingLogsError && Array.isArray(data) ? data : null;
  return [trainingLogs, { mutate, loadingTrainingLogs }, trainingLogsError];
};

export const getTask = async (jobLink) => {
  const res = await fetcher(getConfig()["gatewayBackendUrl"] + jobLink);
  return res;
};
