import { getConfig } from "../config/defaultConfig";
import useSWR from "swr";
import { fetcher } from "./user";

export const useEnvironment = () => {
  const { data, mutate, error } = useSWR(
    getConfig()["environmentAddressesUrl"],
    fetcher
  );
  const loading = !data && !error;
  const environments = data && !error && Array.isArray(data) ? data : null;
  if (environments) {
    environments = environments.map((environment) => {
      return { ...environment, id: environment._id };
    });
  }
  return [environments, { mutate, loading }, error];
};

export const getTask = async (jobLink) => {
  const res = await fetcher(getConfig()["gatewayBackendUrl"] + jobLink);
  return res;
};
