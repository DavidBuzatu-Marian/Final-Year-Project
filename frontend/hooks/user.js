import axios from "axios";
import { getConfig } from "../config/defaultConfig";
import useSWR from "swr";

export const fetcher = (url) =>
  axios
    .get(url, { withCredentials: true })
    .then((res) => res.data)
    .catch((err) => {
      console.log(err);
      if (err) {
        throw err.response.data;
      } else {
        throw Error("Server encountered an error.");
      }
    });

export const useUser = () => {
  const { data, mutate, error } = useSWR(
    getConfig()["authenticatedUrl"],
    fetcher
  );

  const loading = !data && !error;
  const user = data && !error ? data : null;
  return [user, { mutate, loading }, error];
};
